# 本地 Qwen 推理服务 — 对接说明

本文档供其他项目对接本仓库的 `mlx_vlm.server` 时使用。服务是 OpenAI 兼容
接口,无鉴权,只在本机环回口监听,不暴露公网。

---

## 1. 它是什么

| 项 | 值 |
|---|---|
| 软件 | `mlx-vlm` 包提供的 `mlx_vlm.server` |
| 协议 | HTTP + OpenAI Chat Completions 兼容 |
| 模型 | Qwen3.6-35B-A3B-4bit (MoE,文本 + 视觉 + 视频权重) |
| 监听 | `http://127.0.0.1:8080` (硬编码 host=127.0.0.1) |
| 鉴权 | 无 (header `Authorization` 随便填或省略) |
| 并发能力 | 单机单实例,~1 用户流畅,3 用户起明显卡 |

---

## 2. 调用链 — 一次对话怎么走完(6 跳)

不是 RPC,不是共享内存,**就是 HTTP**。OpenAI 兼容协议让客户端 SDK 完全
不用关心后面跑的是 OpenAI 还是本地 mlx-vlm。

```
┌─────────────────────────────────────────────────────────────┐
│ ① 浏览器:用户按 Enter                                       │
│   index.html send()                                          │
│   → 拼 body { messages, tools, stream:true, ... }            │
│   → fetch("/api/chat", { method:"POST" })                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST  application/json
                         ▼ 浏览器 → 127.0.0.1:8090
┌─────────────────────────────────────────────────────────────┐
│ ② Uvicorn(:8090)接管                                        │
│   ASGI 协议 → FastAPI 路由 → server.py 的 `async def chat()`│
│   Pydantic 自动反序列化 body 成 ChatRequest                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ③ Bot 业务逻辑(server.py:chat 函数)                        │
│   - 拼 system prompt (当前时刻 + 联网纪律)                   │
│   - 计算 force_search、tools_for_request                     │
│   - 进入异步生成器 gen()                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ④ ★关键★ 调本地 LLM                                          │
│                                                              │
│   stream = await aclient.chat.completions.create(            │
│       model=MODEL, messages=msgs,                            │
│       tools=tools_for_request, tool_choice="auto",           │
│       stream=True, ... )                                     │
│                                                              │
│   aclient = AsyncOpenAI(                                     │
│       base_url="http://127.0.0.1:8080/v1",                   │
│       api_key="not-needed")                                  │
│                                                              │
│   SDK 内部把它翻译成:                                         │
│     POST http://127.0.0.1:8080/v1/chat/completions           │
│     Content-Type: application/json                           │
│     Accept: text/event-stream  (因为 stream=True)            │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST  内网回环
                         ▼ 127.0.0.1:8090 → 127.0.0.1:8080
┌─────────────────────────────────────────────────────────────┐
│ ⑤ mlx_vlm.server(:8080)                                     │
│   - 把 messages + tools 通过 chat_template.jinja             │
│     拼成模型熟悉的 token 序列(把 tools 翻译成 <tools>...)    │
│   - 调 MLX 框架,GPU/统一内存里逐 token 生成                   │
│   - 每段输出立刻发一行 SSE: `data: {...}\n\n`                 │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP SSE 流  data: 一行一个 chunk
                         ▼ 127.0.0.1:8080 → 127.0.0.1:8090
┌─────────────────────────────────────────────────────────────┐
│ ⑥ Bot 转发 + 回浏览器                                        │
│   async for chunk in stream:                                 │
│     - 收 OpenAI SDK 已解析好的 ChatCompletionChunk           │
│     - 检查 delta.content / delta.tool_calls                  │
│     - 工具调用时执行 (打 Bocha 等)                            │
│     - 把事件包成 SSE 行 yield 给 StreamingResponse           │
│                                                              │
│   index.html reader.read() 循环                              │
│     - JSON.parse 每个 data: 行                               │
│     - bubble.textContent += token                            │
└─────────────────────────────────────────────────────────────┘
```

### 那一行 `aclient.chat.completions.create(...)` 实际上是

```python
# 伪代码,openai SDK 内部做的事
http_body = {
    "model": "/Users/.../Qwen3___6-35B-A3B-4bit",
    "messages": [...],
    "tools": [{"type":"function","function":{...}}, ...],
    "tool_choice": "auto",
    "stream": True,
}
async with httpx.AsyncClient() as c:
    async with c.stream("POST",
        "http://127.0.0.1:8080/v1/chat/completions",
        json=http_body,
        headers={"Authorization":"Bearer not-needed",
                 "Accept":"text/event-stream"}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                yield ChatCompletionChunk.from_json(line[6:])
```

### 为什么这个架构好

- **bot 和 LLM 通过 socket 物理隔离**:改 bot 不用重新加载 21GB 模型;模型挂了 bot 还在
- **base_url 一改就能切后端**:本地 mlx-vlm → 公司内 vLLM 集群 → 云上 DeepSeek,客户端代码零改动
- **工具循环天然就在 bot 层**:模型只负责"决策 + 输出 tool_calls",真正打外部 API 是 bot 干的(打博查 / 打天气 / 打数据库都在 bot 这一层)
- **多对客户端共享同一推理 server**:bot 可以同时有 web UI / 移动 app / 命令行客户端,它们共用 :8080 后端

### 验证这一跳的最快方法

把 bot 关了,直接 curl 模拟跳 ④:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}]}'
```

返回的就是 bot 内部那一行 `aclient.chat.completions.create()` 看到的原始响应。

---

## 3. 启动 / 健康检查

启动(本机):

```bash
cd local-llm
./start_server.sh
```

模型加载 ~4 秒后就绪。验证存活:

```bash
curl -sf http://127.0.0.1:8080/v1/models
# 200 → up;非 2xx → down
```

**重要**:你的项目应该把 server 视为可能随时挂掉的黑盒,每次连接前做健康检查,
失败时给用户明确提示(而不是默默超时)。

---

## 4. API 速查

### GET `/v1/models`
返回已加载模型列表。客户端启动时拉这个拿到 `model_id`,不要在代码里硬编码路径。

```json
{
  "object": "list",
  "data": [
    {
      "id": "/Users/.../Qwen3___6-35B-A3B-4bit",
      "object": "model",
      "created": 1779786638
    }
  ]
}
```

### POST `/v1/chat/completions`
主接口,所有功能(文本/流式/工具/多模态)都走这里。

请求体字段(全部 OpenAI 标准):

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 用 `/v1/models` 里的 id |
| `messages` | array | OpenAI message 数组,见下 |
| `max_tokens` | int | 输出上限,**只限助手回复**,不影响输入 |
| `temperature` | float | 0–2,默认 0.7 |
| `stream` | bool | `true` 走 SSE,见 §6 |
| `tools` | array | function calling 工具列表,见 §7 |
| `tool_choice` | str/obj | `"auto"` / `"required"` / `{"type":"function","function":{"name":"x"}}` |

`messages` 里每条:
```json
{"role": "system" | "user" | "assistant" | "tool", "content": ...}
```
其中 `content` 可以是 **字符串** 或 **content blocks 数组**(多模态,见 §8)。

---

## 5. Python 最小示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

# 启动时拉一次 model id
model_id = client.models.list().data[0].id

# 简单聊天
r = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "用一句话介绍杭州西湖"}],
    max_tokens=200,
)
print(r.choices[0].message.content)
print(f"prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
```

异步版本用 `AsyncOpenAI` 即可,接口完全相同。

---

## 6. 流式(SSE)

```python
stream = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "写一首四句小诗"}],
    stream=True,
    max_tokens=200,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

curl 等价:
```bash
curl -sN -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<id>","messages":[...],"stream":true}'
```

⚠️ **重要 quirk:mlx-vlm 是"伪流式"**。它会发 N 个 `delta.content == ""` 的空 chunk,
然后**最后一个 chunk 一次性塞入完整 content**。表现是"等几秒 → 整段砰一下出现"。
内容正确,但**逐字打字效果丢失**。如果你的 UI 需要打字感,在客户端拿到完整 content
后自己按 30ms/字符重放。

---

## 7. Function Calling (工具调用)

工具 schema 是标准 OpenAI 格式:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市实时天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

r = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "上海今天天气"}],
    tools=tools,
    tool_choice="auto",  # or "required", or {"type":"function","function":{"name":"get_weather"}}
)

msg = r.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(tc.function.name, tc.function.arguments)
        # 执行工具,把结果作为 role=tool 消息追回,再调一次
```

后续轮次要把工具结果按这个格式回传:
```json
{"role": "tool", "tool_call_id": "<原 id>", "content": "<工具返回的 json 字符串>"}
```

⚠️ **重要 quirks**:
1. **forced tool_choice 在长 history 下偶尔不灵** — 模型会绕过工具直接产 content。
   绕过办法:客户端在调用前用关键词路由(`股价/天气/查/搜...`)自己强制走某条路径,
   或者每 N 轮提醒用户新开对话。具体看 `bot/server.py` 的 `FORCE_SEARCH_KEYWORDS`。
2. **模型可能填 enum 之外的字符串** — 比如 `freshness=oneDay` enum,模型可能填
   `"today"`。客户端做 normalize + 兜底,别直接信。
3. **模型可能并行发一堆 tool_calls** — 见过 40+ 次并行同主题查询。设硬上限,例如
   单次请求最多执行 3 次工具(参考 `MAX_TOTAL_TOOL_CALLS`)。

---

## 8. 多模态(图像)

用 OpenAI content blocks 格式:

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = client.chat.completions.create(
    model=model_id,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图里有什么?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    max_tokens=300,
)
print(r.choices[0].message.content)
```

注意:
- `image_url.url` 既可以是 `data:image/...;base64,...`,也可以是公开 http(s) URL
- 一条消息可以放多张图;每多一张图 prefill 慢 ~1s
- 模型有 `vision_cache_size` 默认 20,短时间内重复同一图会命中缓存

PDF 没有原生支持。在 `bot/server.py` 里我们用 PyMuPDF 在客户端侧拆 PDF 为 text + image blocks
再喂给 server,可以直接参考那段代码 (`parse_pdf_to_blocks` 函数)。

---

## 9. JS / 浏览器对接(SSE 流式)

```js
const resp = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: modelId,
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  }),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  let idx;
  while ((idx = buf.indexOf("\n\n")) !== -1) {
    const raw = buf.slice(0, idx);
    buf = buf.slice(idx + 2);
    const line = raw.split("\n").find(l => l.startsWith("data: "));
    if (!line) continue;
    const body = line.slice(6).trim();
    if (body === "[DONE]") break;
    const payload = JSON.parse(body);
    const delta = payload.choices[0].delta;
    if (delta.content) process.stdout.write(delta.content);
  }
}
```

浏览器中如果要从其他端口的页面调 `:8080`(跨域),mlx-vlm 默认允许 CORS,但建议
**在你的项目 backend 做转发代理**,不要让浏览器直接打 :8080 — 这样:
- 隐藏推理 server 的存在
- 可以加 rate limit / auth
- 可以做"伪流式"补救(打字效果重放)

---

## 10. 性能参考(M-series Mac)

| 场景 | 数字 |
|---|---|
| 文本 decode | **~50 tok/s** |
| 单轮 TTFT(热模型,纯文本短输入) | ~200–500ms |
| 单轮 TTFT(图像输入 1 张) | ~2–4s |
| 单轮 TTFT(联网搜索 + 第二轮 prefill) | ~3–5s |
| 模型常驻 GPU/统一内存 | **~21GB** |
| 加载冷启动 | ~4s |
| 并发上限(单机串行) | 1 用户流畅,2–3 用户串行排队 |

部署到普通 CPU 服务器(无 GPU)的话 decode 大约 ~6-15 tok/s,看 CPU 档次。
若部署到 GPU 服务器,建议切到 vLLM 后端,本服务还是单机 Mac 用更优。

---

## 11. 上下文 / 内存边界

- 模型原生上下文 1M token,但 KV cache 占用线性涨。**建议单段对话历史 ≤ 32k token**
- 长对话两个表现:TTFT 越来越慢(prefill 重算);模型容易"重复"或迷路
- 长对话应对:客户端做滑窗(只保留近 N 轮) + 老消息摘要

---

## 12. 常见错误码 / 排错

| 表现 | 大概率原因 |
|---|---|
| connection refused / EOF | server 没起,或 8080 被别的进程占了。`pkill -f mlx_vlm.server` 后重启 |
| 模型加载时 OOM | Mac 内存不够 21GB。关掉其他大应用 |
| 200 但 token 出来很慢 | 内存被 swap 出去了;`footprint -pid <PID>` 看 swapped_out |
| 工具该调没调 | 长 history + forced tool_choice 失灵。让用户新开对话 |
| 流式只收到一个大 chunk | 这是**正常**的"伪流式"行为,不是 bug |
| `400 invalid model` | 用了 hf id 而非本地路径。先打 `/v1/models` 拿 id |

---

## 13. 参考实现 — bot/server.py

本仓库的 `bot/server.py` 是一个完整的"中间层"参考实现,展示了:
- AsyncOpenAI 异步流式转发
- 工具循环(`web_search` + `get_current_time` 两条工具链)
- 多模态 content blocks 透传
- 关键词硬路由防 forced tool_choice 失灵
- Freshness 后端 normalize + 兜底改写
- PDF 解析(文本页抽取 + 扫描页渲染为 image block)

新项目对接时**强烈建议复用以下模式**:
1. 关键词路由(`FORCE_SEARCH_KEYWORDS`)
2. 工具调用次数硬上限(`MAX_TOTAL_TOOL_CALLS`)
3. "诚实交代数据时效"的 system prompt
4. 健康检查端点封装(`/api/health` 模式)
5. `current_time_payload()` 永远注入系统时刻,免得模型瞎猜"今天"

---

## 14. 复用建议清单

- [ ] 客户端启动先打 `/v1/models` 拿 model_id,不要硬编码
- [ ] 健康检查频率 ≥ 15s,挂掉时 UI 明确提示用户
- [ ] 流式响应做"伪流式"重放,弥补 mlx-vlm 不真流式的视觉缺失
- [ ] 工具调用做 normalize + 硬上限,防止模型乱填参数 / 死循环
- [ ] 长对话做滑窗或摘要,别让 history 无限增长
- [ ] 多模态图片做客户端压缩(长边 ≤ 1280px),省 prefill 时间
- [ ] 部署到非 Mac 环境时,切到 llama.cpp 或 vLLM 后端,保持 `BASE_URL` 不变,客户端零改动

---

如果对接遇到本说明没覆盖的边界情况,看 `bot/server.py` 是怎么处理的,
那里基本踩过所有的坑。
