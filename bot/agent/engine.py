"""
Agent 引擎 —— 模型驱动的工具循环,与具体工具/上游解耦。

职责:
  · 跑 "模型 → 工具 → 回写 → 模型" 的多步循环
  · 第 1 轮硬路由(命中关键词强制调某工具),后续轮次交还模型自主决定(auto)
  · 配额封顶(max_steps / max_tool_calls)+ 超额 tool_call 仍回写 tool 消息(协议要求)
  · 流式吐出类型化事件(token / status / tool_start / tool_end / citations / done / error)
  · 上游 client 由外部注入 —— mlx / vLLM / 远程 OpenAI 兼容 API 可任意替换

引擎本身**不认识任何具体工具**,也不组装 system prompt(那是 server 的事)。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi.requests import Request

from .context import ContextManager
from .tools import Tool, ToolContext, ToolResult


@dataclass
class AgentConfig:
    max_steps: int = 3          # 模型<->工具最多来回轮数(模型驱动,留余量让它"看结果再决定")
    max_tool_calls: int = 4     # 单请求总工具执行次数硬上限(无论并行/串行)
    max_tokens: int = 1024
    temperature: float = 0.7


def _evt(type_: str, **kw: Any) -> dict[str, Any]:
    return {"type": type_, **kw}


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        config: AgentConfig | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cfg = config or AgentConfig()
        # 默认带一个上下文管理器(软目标足够高,短对话不会触发,长任务自动折叠旧工具结果)
        self.ctx_mgr = context_manager or ContextManager()

    async def run_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        ctx: ToolContext,
        *,
        request: Request | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """跑一轮完整 agent 循环,逐条 yield SSE 事件(dict)。messages 会被就地追加。"""
        by_name = {t.name: t for t in tools}
        forced = _forced(tools, ctx.user_text)
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        loops = 0
        total_tool_calls = 0
        t0 = time.perf_counter()
        first_t: float | None = None
        chunk_count = 0

        while True:
            # 调模型前整理上下文:越过软预算时折叠较早的大块工具结果(确定性,不靠摘要)
            stat = self.ctx_mgr.manage(messages)
            if stat["folded"]:
                yield _evt("context", folded=stat["folded"],
                           saved=stat["saved"], total=stat["after"])

            allow_tools = (
                bool(tools)
                and loops < self.cfg.max_steps
                and total_tool_calls < self.cfg.max_tool_calls
            )
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            if allow_tools:
                kwargs["tools"] = [t.schema() for t in tools]
                if loops == 0 and forced is not None:
                    kwargs["tool_choice"] = {"type": "function", "function": {"name": forced.name}}
                else:
                    kwargs["tool_choice"] = "auto"

            content_acc = ""
            tool_calls_acc: dict[int, dict[str, str]] = {}
            try:
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if request is not None and await request.is_disconnected():
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        if first_t is None:
                            first_t = time.perf_counter()
                        chunk_count += 1
                        content_acc += delta.content
                        yield _evt("token", text=delta.content)
                    tcs = getattr(delta, "tool_calls", None) if delta else None
                    if tcs:
                        for tc in tcs:
                            idx = getattr(tc, "index", 0) or 0
                            slot = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if getattr(tc, "id", None):
                                slot["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn:
                                if getattr(fn, "name", None):
                                    slot["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    slot["arguments"] += fn.arguments
            except Exception as e:
                yield _evt("error", message=f"模型调用失败: {e}")
                return

            if not tool_calls_acc:
                break  # 模型直接答完

            # 模型要调工具:装 assistant 消息 + 按配额执行 + 回写 tool 消息
            sorted_idxs = sorted(tool_calls_acc.keys())
            remaining_quota = max(0, self.cfg.max_tool_calls - total_tool_calls)
            tool_calls_list: list[dict[str, Any]] = []
            execute_flags: list[bool] = []
            for k, idx in enumerate(sorted_idxs):
                s = tool_calls_acc[idx]
                tool_calls_list.append({
                    "id": s["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": s["name"], "arguments": s["arguments"] or "{}"},
                })
                execute_flags.append(k < remaining_quota)
            messages.append({
                "role": "assistant",
                "content": content_acc or None,
                "tool_calls": tool_calls_list,
            })

            for tc, should_execute in zip(tool_calls_list, execute_flags):
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                if not should_execute:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"已达本次会话工具调用上限 ({self.cfg.max_tool_calls} 次),请基于已有结果直接回答。",
                    })
                    continue

                tool = by_name.get(name)
                if tool is None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"未知工具: {name}",
                    })
                    continue

                # 执行前进度事件:工具自定义优先,否则通用 tool_start
                start_ev = tool.on_start(args)
                if start_ev:
                    yield _evt("status", **start_ev)
                else:
                    yield _evt("tool_start", name=name, args=args)

                try:
                    result = await tool.handler(args, ctx)
                except Exception as e:
                    result = ToolResult(content=f"工具执行失败: {e}")
                    yield _evt("error", message=f"{name} 执行失败: {e}")

                total_tool_calls += 1
                if result.state:
                    ctx.state.update(result.state)
                for ev in result.events:
                    yield ev
                yield _evt("tool_end", name=name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.content,
                })

            loops += 1

        # 收尾:引用 + 统计
        if ctx.citations:
            yield _evt("citations", items=ctx.citations)

        t1 = time.perf_counter()
        if first_t is not None:
            gen_s = max(t1 - first_t, 1e-6)
            stats = {
                "tokens": chunk_count,
                "ttft_ms": round((first_t - t0) * 1000),
                "gen_s": round(gen_s, 2),
                "tps": round(chunk_count / gen_s, 1),
            }
        else:
            stats = {"tokens": 0, "ttft_ms": None, "gen_s": 0, "tps": 0}
        yield _evt("done", stats=stats)


def _forced(tools: list[Tool], user_text: str) -> Tool | None:
    for t in tools:
        if t.force_keywords and any(kw in user_text for kw in t.force_keywords):
            return t
    return None
