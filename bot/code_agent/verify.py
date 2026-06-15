"""
自我验证器 —— 给 Code Agent 的"改完自动检查"闭环提供判定。

默认(零配置):对本轮改动的 .py 文件做语法解析(ast.parse),
抓弱模型最常见的失误——一次坏 edit 把文件改得语法不通。快、通用、无依赖。

可选:传入 verify_cmd(如 'pytest -q' 或 'python -m pytest test_x.py'),
语法通过后再跑它,以退出码判定。覆盖语法之外的逻辑/测试错误。

返回: {"ok": bool, "ran": str, "report": str}
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path


def make_verifier(workdir: str, verify_cmd: str | None = None):
    async def verify(ctx) -> dict:
        edited = sorted(ctx.state.get("edited_files") or [])
        if not edited:
            return {"ok": True, "ran": "(无改动)", "report": ""}

        # ① 语法检查(仅 .py;其他类型跳过,交给 verify_cmd)
        syntax_errors = []
        for f in edited:
            if not f.endswith(".py"):
                continue
            try:
                src = Path(f).read_text("utf-8", errors="replace")
                ast.parse(src, filename=f)
            except SyntaxError as e:
                syntax_errors.append(f"{f}:{e.lineno}: {e.msg}\n    {(e.text or '').rstrip()}")
            except Exception as e:
                syntax_errors.append(f"{f}: 读取/解析失败: {e}")
        if syntax_errors:
            return {
                "ok": False,
                "ran": "语法检查 (ast.parse)",
                "report": "以下文件语法有误:\n" + "\n".join(syntax_errors),
            }

        # ② 可选:跑配置的验证命令
        if verify_cmd:
            try:
                # 禁写 .pyc:避免"编辑后秒级内重复 import 命中陈旧字节码缓存"的假阴性
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
                proc = await asyncio.create_subprocess_shell(
                    verify_cmd, cwd=workdir, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
                text = out.decode("utf-8", errors="replace")
                ok = proc.returncode == 0
                tail = text if len(text) <= 2000 else text[-2000:]
                return {
                    "ok": ok,
                    "ran": verify_cmd,
                    "report": f"exit={proc.returncode}\n{tail}" if not ok else f"exit=0\n{tail[-500:]}",
                }
            except asyncio.TimeoutError:
                return {"ok": False, "ran": verify_cmd, "report": "验证命令超时(>180s)"}
            except Exception as e:
                return {"ok": False, "ran": verify_cmd, "report": f"验证命令执行失败: {e}"}

        return {"ok": True, "ran": "语法检查 (ast.parse)", "report": "语法通过"}

    return verify
