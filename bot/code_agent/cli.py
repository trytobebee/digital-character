"""
本地版 "Claude Code" 的终端 REPL。

用本地 Qwen(mlx_vlm.server @ :8080)+ agent 引擎 + 编码工具,
在终端里读/改代码、跑命令。改文件 / 执行命令前会请你确认。

运行(在 bot/ 目录下):
    ../local-llm/.venv/bin/python -m code_agent
    ../local-llm/.venv/bin/python -m code_agent --workdir /path/to/project

会话内命令:
    /reset   清空对话历史(开新任务,避免历史污染)
    /help    帮助
    /exit    退出(或 Ctrl-D)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from openai import AsyncOpenAI

from agent import Agent, AgentConfig, ToolContext
from .tools import all_coding_tools

# ---------- 终端着色 ----------
C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "red": "\033[31m", "blue": "\033[34m", "mag": "\033[35m",
}


def c(s: str, color: str) -> str:
    return f"{C[color]}{s}{C['reset']}"


SYSTEM_PROMPT = """你是一个在终端里工作的编码助手(类似 Claude Code),运行在用户的真实项目目录中。

你拥有以下工具,必须**通过工具**来观察和修改项目,绝不要凭空猜测文件内容:
- read_file / list_dir / glob_files / grep : 探索和阅读代码
- write_file / edit_file : 修改代码(会请用户确认)
- run_bash : 执行命令(跑测试、git、构建等,会请用户确认)

工作原则:
1. 动手改之前,先用 read_file/grep 看清相关代码的真实内容,基于事实而非记忆。
2. 小改动用 edit_file(精确替换),不要用 write_file 重写整个文件。
3. edit_file 的 old_string 必须和文件内容逐字一致(含缩进);若失败,根据返回的最接近片段修正后重试。
4. 一次只做一件明确的事,做完简要说明你改了什么。
5. 不确定用户意图时,先问清楚再动手,不要擅自大改。
6. 涉及破坏性命令(rm/git push/覆盖文件)务必谨慎,让用户清楚知道后果。
保持简洁,像一个专注的工程师那样工作。"""


class Permission:
    """会话级权限:可对单次操作 y/N,也可 'a' 本会话全部放行。"""

    def __init__(self) -> None:
        self.allow_all = False

    async def confirm(self, action: str, preview: str | None) -> bool:
        if self.allow_all:
            print(c(f"  ↳ {action} (已授权本会话全部操作,自动放行)", "dim"))
            return True
        print()
        print(c(f"  ⚠ {action}", "yellow"))
        if preview:
            for line in preview.splitlines():
                col = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
                print("    " + c(line, col))
        loop = asyncio.get_event_loop()
        ans = (await loop.run_in_executor(None, input, c("    允许? [y/N/a=本会话全允许] ", "bold"))).strip().lower()
        if ans == "a":
            self.allow_all = True
            return True
        return ans in ("y", "yes")


def _resolve_model() -> str:
    disc = Path("/tmp/qwen-local-model-id")
    if disc.exists():
        m = disc.read_text().strip()
        if m:
            return m
    return "/Users/taifeng/code/digital_character/local-llm/models/mlx-community/Qwen3.6-35B-A3B-4bit"


async def run(workdir: str, base_url: str) -> None:
    model = _resolve_model()
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    # 编码任务步数更多,放宽到 8 步 / 12 次工具调用
    agent = Agent(client, model, AgentConfig(max_steps=8, max_tool_calls=12, temperature=0.3))
    tools = all_coding_tools()
    perm = Permission()

    print(c("┌─ 本地 Code Agent ", "cyan") + c(f"(模型 {Path(model).name})", "dim"))
    print(c("│  workdir: ", "cyan") + workdir)
    print(c("│  工具: ", "cyan") + ", ".join(t.name for t in tools))
    print(c("│  /reset 清历史  /ctx 看占用  /help 帮助  /exit 退出", "dim"))
    print(c("└─", "cyan"))

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\n当前 workdir: {workdir}"}]

    while True:
        try:
            user = input(c("\n› ", "bold"))
        except (EOFError, KeyboardInterrupt):
            print(c("\n再见。", "dim"))
            return
        user = user.strip()
        if not user:
            continue
        if user in ("/exit", "/quit"):
            print(c("再见。", "dim"))
            return
        if user == "/help":
            print(c("  直接输入你的需求,比如 '给 server.py 加个 /api/version 接口并跑测试'。", "dim"))
            print(c("  /reset 清空历史开新任务  /exit 退出", "dim"))
            continue
        if user == "/reset":
            history = history[:1]  # 保留 system
            print(c("  已清空对话历史。", "dim"))
            continue
        if user == "/ctx":
            from agent import total_tokens
            tt = total_tokens(history)
            budget = agent.ctx_mgr.cfg.soft_budget
            print(c(f"  当前上下文 ≈ {tt} tok / 软预算 {budget} tok "
                    f"({100*tt//budget}%) · {len(history)} 条消息", "dim"))
            continue

        history.append({"role": "user", "content": user})
        ctx = ToolContext(user_text=user, workdir=workdir, confirm=perm.confirm)

        printed_any = False
        try:
            async for ev in agent.run_stream(history, tools, ctx, max_tokens=1500):
                t = ev["type"]
                if t == "token":
                    if not printed_any:
                        printed_any = True
                    print(ev["text"], end="", flush=True)
                elif t == "tool_start":
                    name = ev.get("name", "?")
                    args = ev.get("args", {})
                    print(c(f"\n● {name}", "mag") + c(f" {_fmt_args(args)}", "dim"))
                elif t == "status" and ev.get("stage") == "searching":
                    print(c(f"\n● 搜索 {ev.get('query','')}", "mag"))
                elif t == "tool_end":
                    pass
                elif t == "context":
                    print(c(f"\n  🗜 折叠了 {ev.get('folded')} 条旧工具结果,省 ~{ev.get('saved')} tok"
                            f"(当前约 {ev.get('total')} tok)", "dim"))
                elif t == "error":
                    print(c(f"\n[错误] {ev.get('message')}", "red"))
                elif t == "done":
                    stats = ev.get("stats", {})
                    print(c(f"\n{C['dim']}  ({stats.get('tokens',0)} tok · "
                            f"{stats.get('ttft_ms','?')}ms 首字 · {stats.get('tps','?')} tok/s){C['reset']}", "dim"))
        except KeyboardInterrupt:
            print(c("\n[已中断本轮]", "yellow"))
            continue
        except Exception as e:
            print(c(f"\n[引擎异常] {e}", "red"))


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return "(" + ", ".join(parts) + ")"


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 LLM 的终端编码 agent")
    ap.add_argument("--workdir", default=None, help="工作根目录(默认当前目录)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="本地模型 OpenAI 兼容端点")
    args = ap.parse_args()
    workdir = str(Path(args.workdir).expanduser().resolve()) if args.workdir else str(Path.cwd())
    try:
        asyncio.run(run(workdir, args.base_url))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
