"""
Agent 引擎闭环自检。

两档:
  离线档(默认,不需要 LLM,不烧任何额度):
    - calculate 工具纯函数测试
    - 用 mock 流式 client 驱动引擎,验证:单工具 / 多工具 / 配额封顶 / 硬路由 / tool 消息回写
  在线档(需要本地 mlx server 在 :8080 跑):
    python test_agent.py live      # 真连 Qwen 跑 calculate(离线工具,不碰博查额度)

用法:
  python test_agent.py            # 离线全套
  python test_agent.py live       # 追加在线一例(需 8080)
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from agent import Agent, AgentConfig, ToolContext, ToolRegistry, builtins


# ---------- 测试替身 ----------
class FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def _delta(**kw):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**kw))])


class ScriptedStream:
    """把一串预设 chunk 当作流返回。"""
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


class ScriptedClient:
    """按调用次数返回不同脚本,模拟 '先要工具 → 再出最终答' 的多轮。"""
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0
        self.seen_kwargs = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.seen_kwargs.append(kwargs)
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return ScriptedStream(script)


def tool_call_chunk(idx, name, args_json):
    tc = SimpleNamespace(index=idx, id=f"call_{idx}", function=SimpleNamespace(name=name, arguments=args_json))
    return _delta(content=None, tool_calls=[tc])


def content_chunk(text):
    return _delta(content=text)


async def collect(agent, msgs, tools, ctx):
    events = []
    async for ev in agent.run_stream(msgs, tools, ctx, request=FakeRequest(), max_tokens=256):
        events.append(ev)
    return events


def types(events):
    return [e["type"] for e in events]


# ---------- 离线用例 ----------
async def test_calculate_pure():
    tool = builtins.make_calculate_tool()
    ctx = ToolContext(user_text="")
    r = await tool.handler({"expression": "(128*3+0.5)/2"}, ctx)
    assert "192.25" in r.content, r.content
    r2 = await tool.handler({"expression": "sqrt(2)*100"}, ctx)
    assert "141.42" in r2.content, r2.content
    r3 = await tool.handler({"expression": "__import__('os')"}, ctx)
    assert "无法计算" in r3.content, r3.content  # 注入被挡
    print("  ✓ calculate 纯函数(含注入防护)")


async def test_single_tool_loop():
    """模型先要 calculate,引擎执行后回写,模型再出最终答。"""
    client = ScriptedClient([
        [tool_call_chunk(0, "calculate", '{"expression": "12*12"}')],
        [content_chunk("结果是 "), content_chunk("144。")],
    ])
    agent = Agent(client, "fake-model", AgentConfig(max_steps=3, max_tool_calls=4))
    tools = [builtins.make_calculate_tool()]
    msgs = [{"role": "user", "content": "12 乘 12 等于几"}]
    ctx = ToolContext(user_text="12 乘 12 等于几")
    ev = await collect(agent, msgs, tools, ctx)
    t = types(ev)
    assert "tool_start" in t and "tool_end" in t, t
    assert "144" in "".join(e.get("text", "") for e in ev if e["type"] == "token")
    # 协议:assistant(带 tool_calls) + tool 消息都回写了
    roles = [m["role"] for m in msgs]
    assert roles.count("assistant") == 1 and roles.count("tool") == 1, roles
    assert msgs[-1]["role"] == "tool" or any(m["role"] == "tool" for m in msgs)
    print("  ✓ 单工具循环(执行→回写→续答),tool 消息协议完整")


async def test_quota_cap():
    """模型一次并行发 6 个工具调用,引擎只执行 max_tool_calls 个,其余回写'已达上限'。"""
    parallel = [tool_call_chunk(i, "calculate", f'{{"expression": "{i}+{i}"}}') for i in range(6)]
    client = ScriptedClient([parallel, [content_chunk("好的。")]])
    agent = Agent(client, "fake-model", AgentConfig(max_steps=3, max_tool_calls=2))
    tools = [builtins.make_calculate_tool()]
    msgs = [{"role": "user", "content": "算一堆"}]
    ctx = ToolContext(user_text="算一堆")
    ev = await collect(agent, msgs, tools, ctx)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 6, f"6 个 tool_call 都要有 tool 消息回写,实际 {len(tool_msgs)}"
    capped = [m for m in tool_msgs if "上限" in m["content"]]
    assert len(capped) == 4, f"应有 4 条'已达上限',实际 {len(capped)}"
    tool_ends = [e for e in ev if e["type"] == "tool_end"]
    assert len(tool_ends) == 2, f"只应真正执行 2 次,实际 {len(tool_ends)}"
    print("  ✓ 配额封顶(执行 2 / 回写 6,超额不执行但协议完整)")


async def test_hard_route():
    """force_keywords 命中 → 第 1 轮强制 tool_choice 指向该工具。"""
    # 造一个带 force_keywords 的假工具(避免触发真博查)
    from agent.tools import Tool, ToolResult

    async def h(args, ctx):
        return ToolResult(content="(命中硬路由)")

    forced_tool = Tool(
        name="demo_search", description="d",
        parameters={"type": "object", "properties": {}},
        handler=h, force_keywords=("最新",),
    )
    client = ScriptedClient([
        [tool_call_chunk(0, "demo_search", "{}")],
        [content_chunk("已答")],
    ])
    agent = Agent(client, "fake-model")
    ctx = ToolContext(user_text="给我最新消息")
    await collect(agent, [{"role": "user", "content": "给我最新消息"}], [forced_tool], ctx)
    first_call_kwargs = client.seen_kwargs[0]
    tc = first_call_kwargs.get("tool_choice")
    assert isinstance(tc, dict) and tc["function"]["name"] == "demo_search", tc
    # 第二轮应回到 auto
    assert client.seen_kwargs[1].get("tool_choice") == "auto", client.seen_kwargs[1].get("tool_choice")
    print("  ✓ 关键词硬路由(首轮强制该工具,次轮回 auto)")


async def test_no_tool_direct_answer():
    """没有工具调用时直接出答,不应有 tool_start/tool_end。"""
    client = ScriptedClient([[content_chunk("你好,"), content_chunk("我是助手。")]])
    agent = Agent(client, "fake-model")
    ctx = ToolContext(user_text="你好")
    ev = await collect(agent, [{"role": "user", "content": "你好"}], [builtins.make_time_tool()], ctx)
    assert "tool_start" not in types(ev)
    assert types(ev)[-1] == "done"
    print("  ✓ 无工具直答(纯文本路径)")


# ---------- 在线用例(需 8080)----------
async def test_live_calculate():
    from openai import AsyncOpenAI
    import os
    from pathlib import Path

    disc = Path("/tmp/qwen-local-model-id")
    model = disc.read_text().strip() if disc.exists() else os.getenv("MODEL", "")
    if not model:
        print("  ! 无 discovery 文件,跳过在线档")
        return
    client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
    agent = Agent(client, model, AgentConfig(max_steps=3, max_tool_calls=3))
    tools = [builtins.make_calculate_tool(), builtins.make_time_tool()]
    q = "我们三个人吃饭花了 386 元,平摊每人多少钱?用计算工具算。"
    msgs = [{"role": "user", "content": q}]
    ctx = ToolContext(user_text=q)
    print(f"  [live] 问: {q}")
    answer = ""
    used_tool = False
    async for ev in agent.run_stream(msgs, tools, ctx, request=FakeRequest(), max_tokens=300):
        if ev["type"] == "tool_start":
            used_tool = True
            print(f"  [live] 模型调工具: {ev.get('name')} {ev.get('args')}")
        elif ev["type"] == "token":
            answer += ev["text"]
        elif ev["type"] == "done":
            print(f"  [live] 答: {answer.strip()[:200]}")
            print(f"  [live] stats: {ev['stats']}")
    print(f"  {'✓' if used_tool else '!'} 在线 calculate(used_tool={used_tool})")


async def main():
    live = len(sys.argv) > 1 and sys.argv[1] == "live"
    print("=== 离线档(无需 LLM)===")
    await test_calculate_pure()
    await test_single_tool_loop()
    await test_quota_cap()
    await test_hard_route()
    await test_no_tool_direct_answer()
    print("离线档全部通过 ✅")
    if live:
        print("\n=== 在线档(需 8080)===")
        await test_live_calculate()


if __name__ == "__main__":
    asyncio.run(main())
