"""
用 OpenAI SDK 访问本地 mlx_lm.server。
跑之前先 `./start_server.sh` 起服务。

涵盖：
  1. list models
  2. 普通 chat
  3. 流式 chat
  4. 带 system prompt 的多轮对话
  5. 工具调用 (function calling)

运行：
  python test_client.py
  python test_client.py 3        # 只跑第 3 个
"""
import json
import sys
import time
from openai import OpenAI

# mlx_lm.server 的 model id 是 --model 传入的本地路径
MODEL = "/Users/taifeng/code/digital_character/local-llm/models/mlx-community/Qwen3___6-35B-A3B-4bit"

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")


def banner(n, title):
    print(f"\n{'='*60}\n[{n}] {title}\n{'='*60}")


def fmt_speed(completion_tokens, elapsed_s, ttft_s=None):
    tps = completion_tokens / elapsed_s if elapsed_s > 0 else 0
    parts = [f"completion={completion_tokens}", f"elapsed={elapsed_s:.2f}s", f"speed={tps:.1f} tok/s"]
    if ttft_s is not None:
        parts.insert(0, f"TTFT={ttft_s*1000:.0f}ms")
    return "[speed] " + "  ".join(parts)


def test_1_list_models():
    banner(1, "list models")
    models = client.models.list()
    for m in models.data:
        print(f"  - {m.id}")


def test_2_simple_chat():
    banner(2, "普通 chat")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话介绍杭州西湖。"}],
        max_tokens=120,
        temperature=0.5,
    )
    elapsed = time.perf_counter() - t0
    print(resp.choices[0].message.content)
    print(f"\n[usage] prompt={resp.usage.prompt_tokens}  "
          f"completion={resp.usage.completion_tokens}")
    print(fmt_speed(resp.usage.completion_tokens, elapsed))


def test_3_streaming():
    banner(3, "流式 chat（token 实时打印）")
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "写一首四句中文小诗，主题是初夏。"}],
        max_tokens=200,
        temperature=0.7,
        stream=True,
    )
    first_t = None
    chunks = 0
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if first_t is None:
                first_t = time.perf_counter()
            chunks += 1
            print(delta.content, end="", flush=True)
    t1 = time.perf_counter()
    print()
    if first_t is not None:
        print(fmt_speed(chunks, t1 - first_t, ttft_s=first_t - t0))


def test_4_multi_turn():
    banner(4, "system prompt + 多轮对话")
    messages = [
        {"role": "system", "content": "你是一名严谨的旅游顾问，回答务必简短（不超过两句）。"},
        {"role": "user", "content": "我下周想去黄山，推荐几日游？"},
    ]
    t0 = time.perf_counter()
    r1 = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=200, temperature=0.4
    )
    e1 = time.perf_counter() - t0
    a1 = r1.choices[0].message.content
    print(f"用户: {messages[-1]['content']}")
    print(f"助手: {a1}")
    print(fmt_speed(r1.usage.completion_tokens, e1))

    messages.append({"role": "assistant", "content": a1})
    messages.append({"role": "user", "content": "山上住宿大概多少钱？"})

    t0 = time.perf_counter()
    r2 = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=200, temperature=0.4
    )
    e2 = time.perf_counter() - t0
    print(f"\n用户: {messages[-1]['content']}")
    print(f"助手: {r2.choices[0].message.content}")
    print(fmt_speed(r2.usage.completion_tokens, e2))


def test_5_function_calling():
    banner(5, "function calling（工具调用）")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询某个城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"},
                        "unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "帮我查一下明天上海要不要带伞？"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=200,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"工具: {tc.function.name}")
            print(f"参数: {tc.function.arguments}")
    else:
        print("模型未调用工具，直接回答:")
        print(msg.content)
    print(fmt_speed(resp.usage.completion_tokens, elapsed))


TESTS = [
    test_1_list_models,
    test_2_simple_chat,
    test_3_streaming,
    test_4_multi_turn,
    test_5_function_calling,
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TESTS[int(sys.argv[1]) - 1]()
    else:
        for t in TESTS:
            t()
