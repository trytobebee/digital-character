"""
本地 Qwen3.6 聊天 Bot 的 FastAPI 后端(HTTP 层)。

工具循环已抽到 agent/ 包:
  agent.engine   模型驱动的工具循环(上游可替换)
  agent.tools    Tool / ToolRegistry 抽象
  agent.builtins get_current_time / calculate / web_search 三个内置工具

本文件只负责:HTTP 路由 + system prompt 组装 + PDF 解析 + 把引擎事件转 SSE。

- GET  /            : 返回 index.html
- POST /api/chat    : SSE 流式聊天(token/status/tool_start/tool_end/citations/done/error)
- POST /api/upload-pdf
- GET  /api/health  : 健康/能力探测
- GET  /api/model   : 对接发现,返回当前 model 字段

启动:
    BOCHA_API_KEY=xxx ./start.sh
"""
from __future__ import annotations

import base64
import json
import os
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

from agent import Agent, AgentConfig, ToolContext, ToolRegistry, builtins

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # 从同目录 .env 读取环境变量(若存在)

# ---------- 配置 ----------
UPSTREAM_BASE_URL = "http://127.0.0.1:8080/v1"

# MODEL 优先从 start_server.sh 写出的发现文件读取(单一真相源,自动对齐),
# 否则回退硬编码。⚠️ 必须和 mlx_vlm.server 的 --model 完全一致,否则触发卸载+重载。
_MODEL_DISCOVERY_FILE = Path("/tmp/qwen-local-model-id")
_MODEL_FALLBACK = "/Users/taifeng/code/digital_character/local-llm/models/mlx-community/Qwen3.6-35B-A3B-4bit"
try:
    MODEL = _MODEL_DISCOVERY_FILE.read_text().strip() or _MODEL_FALLBACK
    print(f"[boot] MODEL from discovery file: {MODEL}", flush=True)
except FileNotFoundError:
    MODEL = _MODEL_FALLBACK
    print(f"[boot] discovery file 不存在,使用 fallback MODEL: {MODEL}", flush=True)

BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "").strip()

PDF_MAX_BYTES = 20 * 1024 * 1024  # 单个 PDF 最大 20MB
PDF_MAX_PAGES = 20                # 一份 PDF 最多处理 20 页
PDF_SPARSE_TEXT_CHARS = 30        # 一页文字少于此阈值视为扫描页,改用图像
PDF_RENDER_DPI = 120              # 扫描页渲染 DPI

# ---------- 引擎 + 工具注册 ----------
aclient = AsyncOpenAI(base_url=UPSTREAM_BASE_URL, api_key="not-needed")
agent_engine = Agent(aclient, MODEL, AgentConfig(max_steps=3, max_tool_calls=4))

# 注册内置工具。加新工具只需在此 register 一行,引擎与 server 主流程都不用改。
registry = ToolRegistry()
registry.register(builtins.make_time_tool())
registry.register(builtins.make_calculate_tool())
registry.register(builtins.make_web_search_tool(lambda: BOCHA_API_KEY))

app = FastAPI()


# ---------- 数据模型 ----------
class Message(BaseModel):
    role: str
    # content 可为字符串(纯文本)或 OpenAI content blocks 数组(多模态)
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    web_search: bool = False


def msg_text(content: str | list[dict[str, Any]]) -> str:
    """从可能多模态的 content 里抽出纯文本(用于日志/决策可读化)。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# ---------- PDF 解析 ----------
def parse_pdf_to_blocks(data: bytes) -> dict[str, Any]:
    """用 PyMuPDF 把 PDF 解析为 OpenAI content blocks:文字页走 text,扫描页渲染 PNG 走 image_url。"""
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
            blocks.append({"type": "text", "text": f"=== 第 {i + 1} 页 / 共 {total_pages} 页 ===\n{text}"})
            text_pages += 1
        else:
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            blocks.append({"type": "text", "text": f"=== 第 {i + 1} 页 / 共 {total_pages} 页 (扫描页,见下图) ==="})
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            image_pages += 1

    doc.close()
    return {
        "content_blocks": blocks,
        "stats": {
            "total_pages": total_pages, "processed": process_count,
            "text_pages": text_pages, "image_pages": image_pages,
            "truncated": total_pages > PDF_MAX_PAGES,
        },
    }


# ---------- system prompt 组装 ----------
def build_system_text(req_system: str | None, use_search: bool) -> str:
    """拼接 system prompt:用户角色 + 当前时刻注入 + (开搜时)诚实交代规则。"""
    parts: list[str] = []
    if req_system and req_system.strip():
        parts.append(req_system.strip())

    now = builtins.current_time_payload()
    parts.append(
        f"当前时刻: {now['friendly']} ({now['timezone']})。"
        "涉及'现在几点/今天日期/星期几/此刻'类系统时间问题时,直接基于此值回答,不需要联网搜索。"
    )
    if use_search:
        parts.append(
            f"当前日期: {now['date']}。涉及实时信息、最新事件、近期数据、或你不确定的具体事实时,优先调用 web_search。"
            "常识/代码/写作类不要调用。一个问题最多调用 3 次,不要并行发相似查询或按类别逐一检索,拿到结果直接综合作答。\n"
            "【对搜索结果的诚实交代】\n"
            "1) 返回 0 条:直说'未在博查索引中找到 <主题> 的相关内容',请用户换问法或稍后再试,不要凭记忆编造。\n"
            "2) 结果全早于今天:明说'我能拿到的最新相关数据停留在 <实际日期>',据此回答,"
            "不要编造'今日休市/节假日/数据未发布'等借口(你不知道现实市场状态)。\n"
            "3) 滞后数据(股价/汇率/天气/赛况):告知'此为最新可检索数据,实时分钟级请到行情软件/官方 APP'。\n"
            "4) 绝不捏造具体数字、价格、温度、比分、人事任命;宁可说'未拿到此数据'。"
        )
    return "\n\n".join(parts).strip()


# ---------- 路由 ----------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
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
    print(f"[pdf] {file.filename!r}  pages={result['stats']['processed']}/{result['stats']['total_pages']}  "
          f"text={result['stats']['text_pages']}  image={result['stats']['image_pages']}", flush=True)
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


@app.get("/api/model")
def get_model() -> JSONResponse:
    return JSONResponse({
        "model": MODEL,
        "discovery_file": str(_MODEL_DISCOVERY_FILE),
        "upstream": UPSTREAM_BASE_URL,
        "tools": [t.name for t in registry.available_tools()],
    })


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    use_search = bool(req.web_search and BOCHA_API_KEY)

    # 组装 messages:system(角色 + 时刻 + 诚实规则)+ 历史
    msgs: list[dict[str, Any]] = []
    sys_text = build_system_text(req.system, use_search)
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    msgs.extend(m.model_dump() for m in req.messages)

    # 工具集:从注册表取可用工具,但 web_search 仅在前端开关打开时才挂(即便有 key)
    tools = [
        t for t in registry.available_tools()
        if t.name != "web_search" or use_search
    ]

    last_user_msg = next((m for m in reversed(req.messages) if m.role == "user"), None)
    last_user_text = msg_text(last_user_msg.content) if last_user_msg else ""
    has_image = bool(
        last_user_msg and isinstance(last_user_msg.content, list)
        and any(isinstance(b, dict) and b.get("type") == "image_url" for b in last_user_msg.content)
    )
    print(
        f"[chat] web_search={req.web_search} tools={[t.name for t in tools]} "
        f"has_image={has_image} msg={last_user_text[:60]!r}", flush=True,
    )

    ctx = ToolContext(user_text=last_user_text)

    async def gen():
        async for ev in agent_engine.run_stream(
            msgs, tools, ctx,
            request=request, max_tokens=req.max_tokens, temperature=req.temperature,
        ):
            yield sse(ev)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
