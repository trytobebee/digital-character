"""
编码 agent 的工具集 —— 让本地模型能在终端里读/改代码、跑命令。

复用 agent.tools 的 Tool/ToolResult/ToolContext 抽象,所以这些工具能直接
挂到同一个 Agent 引擎上。改文件 / 执行命令前通过 ctx.confirm 征求许可。

工具:
  read_file   读文件(带行号,可分段)
  list_dir    列目录
  glob_files  按通配符找文件
  grep        搜文件内容(优先 ripgrep,回退 Python)
  write_file  新建/覆盖文件(需确认,展示 diff)
  edit_file   精确替换(需确认,展示 diff;无匹配时给上下文便于重试)
  run_bash    执行 shell 命令(需确认)
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import shutil
import subprocess
from pathlib import Path

from agent.tools import Tool, ToolContext, ToolResult

MAX_READ_BYTES = 200_000
MAX_OUTPUT_CHARS = 12_000      # 工具回给模型的文本上限,防止撑爆上下文
DEFAULT_READ_LINES = 800


def _resolve(ctx: ToolContext, path: str) -> Path:
    """把工具参数里的路径解析到 workdir 之下(相对路径相对 workdir)。"""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(ctx.workdir) / p
    return p


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... (输出被截断,共 {len(text)} 字符)"


def _unified_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )
    return "".join(diff)


async def _confirm(ctx: ToolContext, action: str, preview: str | None) -> bool:
    """没有 confirm 回调时默认放行(无人值守);有则交给 CLI 询问。"""
    if ctx.confirm is None:
        return True
    return await ctx.confirm(action, preview)


# ---------- read_file ----------
def make_read_file() -> Tool:
    async def handler(args, ctx):
        path = args.get("path", "")
        p = _resolve(ctx, path)
        if not p.exists():
            return ToolResult(content=f"文件不存在: {p}")
        if p.is_dir():
            return ToolResult(content=f"{p} 是目录,请用 list_dir")
        try:
            data = p.read_bytes()[:MAX_READ_BYTES]
            text = data.decode("utf-8", errors="replace")
        except Exception as e:
            return ToolResult(content=f"读取失败: {e}")
        lines = text.splitlines()
        offset = int(args.get("offset", 0) or 0)
        limit = int(args.get("limit", DEFAULT_READ_LINES) or DEFAULT_READ_LINES)
        chunk = lines[offset: offset + limit]
        numbered = "\n".join(f"{offset + i + 1}\t{ln}" for i, ln in enumerate(chunk))
        header = f"# {p}  (第 {offset + 1}-{offset + len(chunk)} 行 / 共 {len(lines)} 行)\n"
        return ToolResult(content=_clip(header + numbered))

    return Tool(
        name="read_file",
        description="读取文本文件内容,返回带行号的文本。大文件用 offset/limit 分段读。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径(相对 workdir 或绝对)"},
                "offset": {"type": "integer", "description": "起始行(0 基),默认 0"},
                "limit": {"type": "integer", "description": "读取行数,默认 800"},
            },
            "required": ["path"],
        },
        handler=handler,
    )


# ---------- list_dir ----------
def make_list_dir() -> Tool:
    async def handler(args, ctx):
        p = _resolve(ctx, args.get("path", "."))
        if not p.exists():
            return ToolResult(content=f"目录不存在: {p}")
        if not p.is_dir():
            return ToolResult(content=f"{p} 不是目录")
        entries = []
        for item in sorted(p.iterdir()):
            if item.name.startswith(".") and item.name not in (".env.example", ".gitignore"):
                continue
            tag = "/" if item.is_dir() else ""
            size = "" if item.is_dir() else f"  ({item.stat().st_size}B)"
            entries.append(f"{item.name}{tag}{size}")
        return ToolResult(content=_clip(f"# {p}\n" + "\n".join(entries)))

    return Tool(
        name="list_dir",
        description="列出目录下的文件和子目录(忽略多数隐藏项)。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "目录路径,默认当前 workdir"}},
            "required": [],
        },
        handler=handler,
    )


# ---------- glob_files ----------
def make_glob() -> Tool:
    async def handler(args, ctx):
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult(content="缺少 pattern")
        root = Path(ctx.workdir)
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules", "models")]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern):
                    matches.append(rel)
        matches.sort()
        if not matches:
            return ToolResult(content=f"无匹配 {pattern!r} 的文件")
        return ToolResult(content=_clip(f"# glob {pattern!r}  ({len(matches)} 个)\n" + "\n".join(matches[:300])))

    return Tool(
        name="glob_files",
        description="按通配符在 workdir 内递归找文件(如 '**/*.py'、'server.py')。",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "通配符,如 *.py 或 bot/**/*.html"}},
            "required": ["pattern"],
        },
        handler=handler,
    )


# ---------- grep ----------
def make_grep() -> Tool:
    has_rg = shutil.which("rg") is not None

    async def handler(args, ctx):
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult(content="缺少 pattern")
        path = args.get("path") or "."
        target = _resolve(ctx, path)
        if has_rg:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "rg", "-n", "--no-heading", "-S", "--max-count", "200", pattern, str(target),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                out, err = await proc.communicate()
                text = out.decode("utf-8", errors="replace")
                if not text.strip():
                    return ToolResult(content=f"无匹配 {pattern!r}")
                return ToolResult(content=_clip(text))
            except Exception:
                pass  # 回退 Python
        # Python 回退:逐文件扫
        import re as _re
        try:
            rx = _re.compile(pattern)
        except _re.error as e:
            return ToolResult(content=f"正则错误: {e}")
        hits = []
        roots = [target] if target.is_file() else [target]
        for dirpath, dirnames, filenames in os.walk(roots[0] if roots[0].is_dir() else roots[0].parent):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "node_modules", "models")]
            for fn in filenames:
                fp = Path(dirpath) / fn
                if target.is_file() and fp != target:
                    continue
                try:
                    for i, ln in enumerate(fp.read_text("utf-8", errors="ignore").splitlines(), 1):
                        if rx.search(ln):
                            hits.append(f"{fp}:{i}:{ln.strip()[:200]}")
                            if len(hits) >= 200:
                                break
                except Exception:
                    continue
        return ToolResult(content=_clip("\n".join(hits)) if hits else f"无匹配 {pattern!r}")

    return Tool(
        name="grep",
        description="在文件/目录中按正则搜索内容,返回 文件:行号:内容。优先用 ripgrep。",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式"},
                "path": {"type": "string", "description": "搜索路径,默认 workdir"},
            },
            "required": ["pattern"],
        },
        handler=handler,
    )


# ---------- write_file ----------
def make_write_file() -> Tool:
    async def handler(args, ctx):
        path = args.get("path", "")
        content = args.get("content", "")
        p = _resolve(ctx, path)
        old = p.read_text("utf-8", errors="replace") if p.exists() else ""
        diff = _unified_diff(old, content, str(p)) or "(新文件)\n" + content[:1000]
        action = f"写入文件 {p}" + (" (覆盖)" if old else " (新建)")
        if not await _confirm(ctx, action, diff):
            return ToolResult(content="用户拒绝了写入操作")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=f"写入失败: {e}")
        ctx.state.setdefault("edited_files", set()).add(str(p))
        return ToolResult(content=f"已写入 {p} ({len(content)} 字符)")

    return Tool(
        name="write_file",
        description="新建或覆盖整个文件。会先展示 diff 并请用户确认。小改动优先用 edit_file。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "完整文件内容"},
            },
            "required": ["path", "content"],
        },
        handler=handler,
    )


# ---------- edit_file ----------
def make_edit_file() -> Tool:
    async def handler(args, ctx):
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = bool(args.get("replace_all", False))
        p = _resolve(ctx, path)
        if not p.exists():
            return ToolResult(content=f"文件不存在: {p}(新建请用 write_file)")
        text = p.read_text("utf-8", errors="replace")
        if old_string == new_string:
            return ToolResult(content="old_string 与 new_string 相同,无需修改")
        count = text.count(old_string)
        if count == 0:
            # 给出最相近的几行,帮助模型修正 old_string
            hint = _closest_lines(text, old_string)
            return ToolResult(content=f"未找到 old_string。文件中最接近的片段:\n{hint}\n请据实际内容修正 old_string 后重试。")
        if count > 1 and not replace_all:
            return ToolResult(content=f"old_string 在文件中出现 {count} 次,不唯一。请扩大 old_string 使其唯一,或设 replace_all=true。")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        diff = _unified_diff(text, new_text, str(p))
        if not await _confirm(ctx, f"编辑文件 {p}", diff):
            return ToolResult(content="用户拒绝了编辑操作")
        try:
            p.write_text(new_text, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=f"写入失败: {e}")
        ctx.state.setdefault("edited_files", set()).add(str(p))
        return ToolResult(content=f"已编辑 {p}(替换 {count if replace_all else 1} 处)")

    return Tool(
        name="edit_file",
        description=(
            "对已有文件做精确字符串替换。old_string 必须与文件内容逐字匹配(含缩进)且唯一。"
            "会先展示 diff 并请用户确认。匹配失败会返回最接近的片段供你修正。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_string": {"type": "string", "description": "要替换的原文(逐字、唯一)"},
                "new_string": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {"type": "boolean", "description": "是否替换所有出现,默认 false"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=handler,
    )


def _closest_lines(text: str, needle: str, n: int = 6) -> str:
    """在文件里找与 needle 首行最相似的几行,作为修正提示。"""
    first = (needle.strip().splitlines() or [""])[0]
    lines = text.splitlines()
    scored = sorted(
        ((difflib.SequenceMatcher(None, first, ln).ratio(), i, ln) for i, ln in enumerate(lines, 1)),
        reverse=True,
    )[:n]
    scored.sort(key=lambda x: x[1])
    return "\n".join(f"{i}\t{ln}" for _, i, ln in scored)


# ---------- run_bash ----------
def make_run_bash() -> Tool:
    async def handler(args, ctx):
        cmd = args.get("command", "")
        if not cmd.strip():
            return ToolResult(content="缺少 command")
        if not await _confirm(ctx, "执行命令", f"$ {cmd}"):
            return ToolResult(content="用户拒绝了命令执行")
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=ctx.workdir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(content="命令超时(>120s)已终止")
        except Exception as e:
            return ToolResult(content=f"执行失败: {e}")
        text = out.decode("utf-8", errors="replace")
        return ToolResult(content=_clip(f"exit={proc.returncode}\n{text}"))

    return Tool(
        name="run_bash",
        description=(
            "在 workdir 下执行一条 shell 命令并返回 stdout/stderr 与退出码。"
            "用于跑测试、git、构建、查看环境等。会先请用户确认。超时 120s。"
        ),
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "完整 shell 命令"}},
            "required": ["command"],
        },
        handler=handler,
    )


def all_coding_tools() -> list[Tool]:
    return [
        make_read_file(), make_list_dir(), make_glob(), make_grep(),
        make_write_file(), make_edit_file(), make_run_bash(),
    ]
