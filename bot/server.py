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

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
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

PDF_MAX_BYTES = 20 * 1024 * 1024  # 单个 PDF 最大 20MB
PDF_MAX_PAGES = 20                # 一份 PDF 最多处理 20 页
PDF_SPARSE_TEXT_CHARS = 30        # 一页文字少于此阈值视为扫描页,改用图像
PDF_RENDER_DPI = 120              # 扫描页渲染 DPI(质量 vs 体积平衡)

# 当用户消息含以下"严格今天"词时,后端把 freshness 兜底改写成今日 YYYY-MM-DD,
# 收紧博查的过去 24 小时滚动窗口(避免拿到昨天的内容)
TODAY_STRICT_KEYWORDS = ("今天", "此刻", "现在", "刚刚", "刚才", "今晨", "今早", "今晚", "今夜")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRESHNESS_ENUMS = {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}


def normalize_freshness(fr: str | None) -> str:
    """模型现在可以填精确日期了,做一层后端 validate,非法值兜底为 noLimit"""
    if not fr:
        return "noLimit"
    if fr in _FRESHNESS_ENUMS:
        return fr
    if _DATE_RE.match(fr):
        return fr
    return "noLimit"


def maybe_override_to_today(fr: str, user_text: str) -> str:
    """方案 B:用户说'今天/此刻/现在/刚刚'时,把 oneDay/noLimit 收紧到精确今日 YYYY-MM-DD"""
    if fr in ("oneDay", "noLimit") and any(kw in user_text for kw in TODAY_STRICT_KEYWORDS):
        return time.strftime("%Y-%m-%d")
    return fr


# 命中以下关键词时,第 1 轮强制 tool_choice=web_search,绕过模型 RLHF 反射 + 历史污染
# (后续轮次仍然 auto,让模型自己决定要不要二次搜索 / 直接答)
FORCE_SEARCH_KEYWORDS = (
    # 显式搜索指令
    "搜索", "搜一下", "搜下", "查一下", "查询", "查最新", "帮我查", "联网",
    # 实时数据
    "股价", "股票", "市值", "汇率", "天气", "温度", "价格", "行情", "比分",
    # 时间限定词
    "今天", "今早", "今晚", "今夜", "此刻", "现在的", "目前", "刚刚", "刚才",
    "最近", "最新", "本周", "本月", "本年", "近期", "近况", "当下",
    # 训练截止后的年份
    "2025", "2026", "2027",
    # 新闻类信号
    "新闻", "宣布", "新发布", "刚上线", "刚上市",
)

aclient = AsyncOpenAI(base_url=UPSTREAM_BASE_URL, api_key="not-needed")
app = FastAPI()


# ---------- Tool schema ----------
TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "通过博查搜索引擎从主流公开网站(新闻、财经、百科、论文、官方站点等)"
            "检索当前可获取的公开信息。这是检索公开数据,不是给出投资/医疗/法律建议,"
            "你只是把搜索结果转述给用户。\n\n"
            "必须调用的情况(不允许以'我无法访问实时数据''我不能提供投资建议'等理由拒答):\n"
            "- 用户明确要求'搜索/联网/查一下/查询/帮我查'\n"
            "- 涉及实时或近实时数据:股价、市值、汇率、商品价格、天气、赛事比分、航班/列车状态\n"
            "- 涉及训练截止之后的事件:2025 年之后的新闻、发布、人事、政策、论文、产品\n"
            "- 涉及'今天/最新/最近/本周/本月/刚刚/此刻/现在的'等时间限定的事实性问题\n"
            "- 用户提及你不熟悉的具体公司/产品/人物/论文,需要核实事实\n\n"
            "不应调用的情况:\n"
            "- 纯写作任务(写诗、起草邮件、翻译、扩写)\n"
            "- 代码/算法/数学解释\n"
            "- 静态常识(历史事件、基础概念、定义)\n"
            "- 用户的主观偏好/意见请求\n\n"
            "调用纪律:\n"
            "1) 一次提问最多调用 1-2 次本工具。\n"
            "2) 不要并行发起多个相似查询,也不要按类别(政治/经济/科技/...)逐一检索。\n"
            "3) 使用一个综合、足够具体的查询词;一次结果不充分时,再换关键词重搜一次,到此为止。\n"
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
                    "description": (
                        "时间范围过滤。可选值有两类:\n"
                        "(a) 枚举: noLimit / oneDay / oneWeek / oneMonth / oneYear\n"
                        "(b) 精确日期: YYYY-MM-DD 格式 (如 2026-05-27)\n"
                        "选用建议:\n"
                        "- '今天/此刻/现在/刚刚' → 用今日的 YYYY-MM-DD (绝不要用 oneDay,后者是过去 24 小时滚动窗口,会含昨天内容)\n"
                        "- '本周/最近几天' → oneWeek\n"
                        "- '本月/最近一个月' → oneMonth\n"
                        "- '今年' → oneYear\n"
                        "- 不确定就用 noLimit"
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
    # content 可以是字符串 (纯文本) 或 OpenAI content blocks 数组 (多模态):
    #   [{"type":"text","text":"..."},
    #    {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    web_search: bool = False


def msg_text(content: str | list[dict[str, Any]]) -> str:
    """从可能是多模态的 content 里抽出纯文本(用于日志/搜索决策的可读化)"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


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


# ---------- PDF 解析 ----------
def parse_pdf_to_blocks(data: bytes) -> dict[str, Any]:
    """
    用 PyMuPDF 把 PDF 解析为 OpenAI content blocks。
    每页有正文 (>= PDF_SPARSE_TEXT_CHARS 字符) 走 text block;
    稀疏/扫描页走 image_url block(渲染为 PNG)。

    返回:
      content_blocks: list[dict] — 可直接拼到 user message 的 content
      stats: 文件元信息(总页数 / 文本页数 / 图像页数 / 是否截断)
    """
    doc = fitz.open(stream=data, filetype="pdf")
    total_pages = doc.page_count
    process_count = min(total_pages, PDF_MAX_PAGES)

    blocks: list[dict[str, Any]] = []
    text_pages = 0
    image_pages = 0

    for i in range(process_count):
        page = doc[i]
        text = (page.get_text() or "").strip()
        if len(text) >= PDF_SPARSE_TEXT_CHARS:
            blocks.append({
                "type": "text",
                "text": f"=== 第 {i + 1} 页 / 共 {total_pages} 页 ===\n{text}",
            })
            text_pages += 1
        else:
            # 渲染为 PNG 走 VL 路径
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode()
            blocks.append({
                "type": "text",
                "text": f"=== 第 {i + 1} 页 / 共 {total_pages} 页 (扫描页,见下图) ===",
            })
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
            image_pages += 1

    doc.close()
    return {
        "content_blocks": blocks,
        "stats": {
            "total_pages": total_pages,
            "processed": process_count,
            "text_pages": text_pages,
            "image_pages": image_pages,
            "truncated": total_pages > PDF_MAX_PAGES,
        },
    }


# ---------- 路由 ----------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """接收 PDF 文件,返回 content_blocks(text + image_url 混合)+ stats。"""
    if not (file.filename or "").lower().endswith(".pdf") and file.content_type != "application/pdf":
        return JSONResponse({"error": "仅支持 .pdf 文件"}, status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "文件为空"}, status_code=400)
    if len(data) > PDF_MAX_BYTES:
        return JSONResponse(
            {"error": f"PDF 过大 ({len(data)/1024/1024:.1f}MB),上限 {PDF_MAX_BYTES//1024//1024}MB"},
            status_code=400,
        )
    try:
        result = parse_pdf_to_blocks(data)
    except Exception as e:
        return JSONResponse({"error": f"PDF 解析失败: {e}"}, status_code=400)
    result["filename"] = file.filename
    print(f"[pdf] {file.filename!r}  pages={result['stats']['processed']}/{result['stats']['total_pages']}  text={result['stats']['text_pages']}  image={result['stats']['image_pages']}", flush=True)
    return JSONResponse(result)


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
            "拿到结果后直接综合回答,不要为了'更全面'反复重搜。\n\n"
            "【重要:对搜索结果的诚实交代】\n"
            "1) 搜索返回 0 条结果:直接告诉用户'未在博查索引中找到 <主题> 的相关内容',"
            "可能原因是该话题暂未被收录或关键词需要调整,请用户换种问法或稍后再试。"
            "不要凭训练记忆编造答案。\n"
            "2) 搜索结果全部早于今天:明确说出'我能拿到的最新相关数据停留在 <实际日期>',"
            "并据此回答,不要编造'市场未开盘''今日休市''节假日''数据尚未发布'等借口"
            f"(模型不知道现实的市场状态,不要瞎猜)。\n"
            "3) 搜索结果是滞后数据:对实时性敏感的查询(股价/汇率/天气/赛况),"
            "明确告知用户'此为最新可检索的数据,实时分钟级行情请到行情软件/官方 APP 查看'。\n"
            "4) 绝对不要捏造具体数字、价格、温度、比分、人事任命等事实。"
            "宁可说'未拿到此数据',也不要给假数据。"
        )
        sys_text = (sys_text + "\n\n" + extra).strip() if sys_text else extra
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs.extend(m.model_dump() for m in req.messages)

    use_tools = bool(req.web_search and BOCHA_API_KEY)
    # 诊断日志:显示请求关键字段,排查 UI 联网开关是否真的发了上来
    last_user_msg = next((m for m in reversed(req.messages) if m.role == "user"), None)
    last_user_text = msg_text(last_user_msg.content) if last_user_msg else ""
    has_image = isinstance(last_user_msg.content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url"
        for b in last_user_msg.content
    ) if last_user_msg else False
    # 关键词硬路由:命中即强制第 1 轮调 web_search,绕过模型 RLHF 反射 + 历史污染
    matched_kws = [kw for kw in FORCE_SEARCH_KEYWORDS if kw in last_user_text]
    force_search = use_tools and bool(matched_kws)
    print(
        f"[chat] web_search={req.web_search}  use_tools={use_tools}  "
        f"force_search={force_search}  matched_kws={matched_kws[:5]}  "
        f"has_image={has_image}  msg={last_user_text[:60]!r}",
        flush=True,
    )

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
            print(f"[chat]   loop={loops} ENTRY allow_tools={allow_tools} total_calls={total_tool_calls}", flush=True)
            kwargs: dict[str, Any] = dict(
                model=MODEL,
                messages=msgs,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
            )
            if allow_tools:
                kwargs["tools"] = TOOLS_SPEC
                if loops == 0 and force_search:
                    # 第 1 轮强制调 web_search;后续轮次回到 auto 让模型自己决定
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "web_search"},
                    }
                else:
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
                print(f"[chat]   loop={loops} no tool_calls, content_len={len(content_acc)}", flush=True)
                break  # 模型直接回答完毕
            print(f"[chat]   loop={loops} got {len(tool_calls_acc)} tool_calls", flush=True)

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
                    fr_raw = args.get("freshness", "noLimit") or "noLimit"
                    fr = normalize_freshness(fr_raw)
                    fr_after = maybe_override_to_today(fr, last_user_text)
                    if fr_after != fr:
                        print(f"[chat]   freshness override: {fr_raw!r} -> {fr_after!r}", flush=True)
                    fr = fr_after
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
