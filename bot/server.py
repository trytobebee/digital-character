"""
本地 Qwen3.6 聊天 Bot 的 FastAPI 后端。

- GET  /            : 返回 index.html
- POST /api/chat    : SSE 流式聊天,事件类型:
    - token       {text}
    - status      {stage, query, freshness}   联网搜索进度
    - citations   {items:[{index,title,url,site,date}]}
    - done        {stats}
    - error       {message}
- GET  /api/health  : 健康/能力探测

启动:
    BOCHA_API_KEY=xxx ./start.sh
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # 从同目录 .env 文件读取环境变量(若存在)

# ---------- 配置 ----------
UPSTREAM_BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "/Users/taifeng/code/digital_character/local-llm/models/mlx-community/Qwen3___6-35B-A3B-4bit"

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "").strip()
BOCHA_URL = "https://api.bochaai.com/v1/web-search"

MAX_TOOL_LOOPS = 2          # 最多与模型来回 2 轮 tool 协商
MAX_TOTAL_TOOL_CALLS = 3    # 单次请求总共最多执行的 web_search 次数(无论并行还是串行)
SEARCH_COUNT = 8            # 每次 web_search 返回结果数(给模型更多上下文,减少它重搜的动机)
SEARCH_TIMEOUT_S = 15.0

aclient = AsyncOpenAI(base_url=UPSTREAM_BASE_URL, api_key="not-needed")
app = FastAPI()


# ---------- Tool schema ----------
TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网搜索获取最新的网页信息。当用户问题涉及实时/最新/近期事件、"
            "当前数据、新发布的内容,或你不确定的具体事实时,调用此工具。"
            "对常识问题、写作任务、代码问题不要调用。\n"
            "重要约束:\n"
            "1) 对一个用户问题最多调用 1-2 次。\n"
            "2) 不要并行发起多个相似查询,也不要按'类别'(政治/经济/科技/...)分别检索。\n"
            "3) 使用一个综合、足够具体的查询词即可;若一次搜索结果不充分,再换关键词重搜一次,最多到此为止。\n"
            "4) 拿到结果后直接基于结果作答,不要为了'更全面'反复检索。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "简洁的搜索关键词,使用更接近搜索引擎的形式,不要带'请帮我'之类的口语",
                },
                "freshness": {
                    "type": "string",
                    "enum": ["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"],
                    "description": (
                        "时间范围过滤:'今天/最新/刚刚/此刻'用 oneDay;'本周/这几天'用 oneWeek;"
                        "'本月/最近一个月'用 oneMonth;'今年'用 oneYear;不确定就用 noLimit。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}]


# ---------- 数据模型 ----------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    web_search: bool = False


# ---------- Bocha 搜索 ----------
async def bocha_search(query: str, freshness: str = "noLimit") -> dict[str, Any]:
    if not BOCHA_API_KEY:
        raise RuntimeError("BOCHA_API_KEY 未设置")
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S) as c:
        r = await c.post(
            BOCHA_URL,
            headers={
                "Authorization": f"Bearer {BOCHA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"query": query, "freshness": freshness, "summary": True, "count": SEARCH_COUNT},
        )
        if r.status_code >= 400:
            # 把博查返回的响应体带上,方便定位 invalid key / quota 等问题
            body = r.text[:400]
            raise RuntimeError(f"HTTP {r.status_code} from Bocha: {body}")
        return r.json()


def format_search_results(payload: dict[str, Any], offset: int) -> tuple[str, list[dict]]:
    """把 Bocha 响应转成 (给模型读的纯文本, 引用列表)。"""
    data = payload.get("data") or {}
    pages = ((data.get("webPages") or {}).get("value")) or []
    citations: list[dict] = []
    parts: list[str] = []
    for i, p in enumerate(pages, 1):
        idx = offset + i
        title = (p.get("name") or "").strip()
        url = (p.get("url") or "").strip()
        site = (p.get("siteName") or "").strip()
        date = (p.get("datePublished") or "")[:10]
        summary = (p.get("summary") or p.get("snippet") or "").strip()[:800]
        citations.append({
            "index": idx, "title": title, "url": url, "site": site, "date": date,
        })
        parts.append(
            f"[{idx}] {title}\n来源: {site}{(' · ' + date) if date else ''}\n{summary}\nURL: {url}"
        )
    text = "\n\n".join(parts) if parts else "(无搜索结果)"
    return text, citations


# ---------- 路由 ----------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/health")
async def health() -> JSONResponse:
    info = {"model": False, "search": bool(BOCHA_API_KEY)}
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{UPSTREAM_BASE_URL}/models")
            info["model"] = r.status_code == 200
    except Exception:
        pass
    return JSONResponse(info)


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    msgs: list[dict[str, Any]] = []
    if req.system and req.system.strip():
        sys_text = req.system.strip()
    else:
        sys_text = ""
    # 若开了联网,把时间常识塞进 system,模型才知道"今天"是哪天
    if req.web_search and BOCHA_API_KEY:
        today = time.strftime("%Y-%m-%d")
        extra = (
            f"当前日期: {today}。当问题涉及实时信息、最新事件、近期数据、"
            "或你不确定的具体事实时,优先调用 web_search 工具。常识/代码/写作类问题不要调用。"
            f"对一个用户问题最多调用 {MAX_TOTAL_TOOL_CALLS} 次 web_search,"
            "不要并行发起多个相似查询,也不要按类别(政治/经济/科技/...)逐一检索。"
            "拿到结果后直接综合回答,不要为了'更全面'反复重搜。"
        )
        sys_text = (sys_text + "\n\n" + extra).strip() if sys_text else extra
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs.extend(m.model_dump() for m in req.messages)

    use_tools = bool(req.web_search and BOCHA_API_KEY)

    async def gen():
        all_citations: list[dict] = []
        loops = 0
        total_tool_calls = 0
        t0 = time.perf_counter()
        first_t: float | None = None
        chunk_count = 0

        while True:
            # 同时受 loop 次数和总调用次数限制
            allow_tools = (
                use_tools
                and loops < MAX_TOOL_LOOPS
                and total_tool_calls < MAX_TOTAL_TOOL_CALLS
            )
            kwargs: dict[str, Any] = dict(
                model=MODEL,
                messages=msgs,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
            )
            if allow_tools:
                kwargs["tools"] = TOOLS_SPEC
                kwargs["tool_choice"] = "auto"

            content_acc = ""
            tool_calls_acc: dict[int, dict[str, str]] = {}
            try:
                stream = await aclient.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if await request.is_disconnected():
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        if first_t is None:
                            first_t = time.perf_counter()
                        chunk_count += 1
                        content_acc += delta.content
                        yield sse({"type": "token", "text": delta.content})
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
                yield sse({"type": "error", "message": f"模型调用失败: {e}"})
                return

            if not tool_calls_acc:
                break  # 模型直接回答完毕

            # 模型要调工具:构造 assistant 消息 + 执行 + tool 消息
            # 单轮内若并行发了过多 tool_calls,只执行剩余配额数那么多;但每个 tool_call 仍需对应 tool 消息回写
            sorted_idxs = sorted(tool_calls_acc.keys())
            remaining_quota = max(0, MAX_TOTAL_TOOL_CALLS - total_tool_calls)
            tool_calls_list = []
            execute_flags: list[bool] = []
            for k, idx in enumerate(sorted_idxs):
                s = tool_calls_acc[idx]
                tool_calls_list.append({
                    "id": s["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": s["name"], "arguments": s["arguments"] or "{}"},
                })
                execute_flags.append(k < remaining_quota)
            msgs.append({
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
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"已达到本次会话的搜索调用上限 ({MAX_TOTAL_TOOL_CALLS} 次),请基于已有搜索结果直接回答用户。",
                    })
                    continue

                if name == "web_search":
                    q = args.get("query", "") or ""
                    fr = args.get("freshness", "noLimit") or "noLimit"
                    yield sse({"type": "status", "stage": "searching", "query": q, "freshness": fr})
                    try:
                        payload = await bocha_search(q, fr)
                        text, cits = format_search_results(payload, offset=len(all_citations))
                        all_citations.extend(cits)
                        tool_result = text
                    except Exception as e:
                        tool_result = f"搜索失败: {e}"
                        yield sse({"type": "error", "message": f"web_search 失败: {e}"})
                    total_tool_calls += 1
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
                else:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"未知工具: {name}",
                    })

            loops += 1
            # 下一轮:把搜索结果带回模型,要么继续搜(loops < MAX),要么直接答(超 MAX 时禁用 tools)

        # 收尾
        if all_citations:
            yield sse({"type": "citations", "items": all_citations})

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
        yield sse({"type": "done", "stats": stats})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
