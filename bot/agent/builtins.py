"""
内置工具实现。每个工厂函数返回一个 Tool,server 在启动时注册进 ToolRegistry。

目前:
  get_current_time  本机时刻(无网络依赖,永远可用)
  calculate         安全四则/数学表达式求值(纯离线,可不烧外部额度做闭环测试)
  web_search        博查联网检索(需 BOCHA_API_KEY;含 freshness 兜底改写)
"""
from __future__ import annotations

import ast
import math
import operator
import re
import time
from typing import Any

import httpx

from .tools import Tool, ToolContext, ToolResult

WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def current_time_payload() -> dict[str, Any]:
    """本机当前时间信息(供工具返回 + system prompt 注入复用)。"""
    lt = time.localtime()
    wd = WEEKDAY_ZH[lt.tm_wday]
    return {
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", lt),
        "date": time.strftime("%Y-%m-%d", lt),
        "time": time.strftime("%H:%M:%S", lt),
        "weekday": wd,
        "timezone": time.strftime("%Z", lt) or "本机时区",
        "friendly": time.strftime("%Y年%m月%d日 ", lt) + wd + time.strftime(" %H:%M:%S", lt),
    }


# ---------- get_current_time ----------
def make_time_tool() -> Tool:
    async def handler(args: dict, ctx: ToolContext) -> ToolResult:
        import json
        return ToolResult(content=json.dumps(current_time_payload(), ensure_ascii=False))

    return Tool(
        name="get_current_time",
        description=(
            "获取服务器当前时刻(本机时区,通常即中国标准时间 UTC+8)。"
            "返回 ISO 时间戳、日期、时间、星期、时区。\n"
            "用户问'现在几点''今天日期''今天星期几'等系统时间问题时调用,确保是实时时间而非训练记忆猜测。"
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


# ---------- calculate(纯离线,证明可插拔)----------
_ALLOWED_BINOP = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    "max": max, "min": min, "pow": pow, "exp": math.exp,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOP:
        return _ALLOWED_BINOP[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_FUNCS and not callable(_ALLOWED_FUNCS[node.id]):
        return _ALLOWED_FUNCS[node.id]  # pi / e
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        fn = _ALLOWED_FUNCS[node.func.id]
        return fn(*[_safe_eval(a) for a in node.args])
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def make_calculate_tool() -> Tool:
    async def handler(args: dict, ctx: ToolContext) -> ToolResult:
        expr = (args.get("expression") or "").strip()
        if not expr:
            return ToolResult(content="错误: 缺少 expression")
        try:
            tree = ast.parse(expr, mode="eval")
            val = _safe_eval(tree)
            return ToolResult(content=f"{expr} = {val}")
        except Exception as e:
            return ToolResult(content=f"无法计算 {expr!r}: {e}")

    return Tool(
        name="calculate",
        description=(
            "计算一个数学表达式并返回精确结果。支持 + - * / // % **、括号、"
            "sqrt/abs/round/log/exp/sin/cos/tan/max/min/pow 及 pi/e。\n"
            "涉及数值计算(账单分摊、折扣、面积、利息、单位换算的乘除等)时调用,"
            "不要心算,以本工具结果为准。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "纯数学表达式,如 '(128*3+0.5)/2' 或 'sqrt(2)*100'"},
            },
            "required": ["expression"],
        },
        handler=handler,
    )


# ---------- web_search(博查)----------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRESHNESS_ENUMS = {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}
TODAY_STRICT_KEYWORDS = ("今天", "此刻", "现在", "刚刚", "刚才", "今晨", "今早", "今晚", "今夜")

FORCE_SEARCH_KEYWORDS = (
    "搜索", "搜一下", "搜下", "查一下", "查询", "查最新", "帮我查", "联网",
    "股价", "股票", "市值", "汇率", "天气", "温度", "价格", "行情", "比分",
    "今天", "今早", "今晚", "今夜", "此刻", "现在的", "目前", "刚刚", "刚才",
    "最近", "最新", "本周", "本月", "本年", "近期", "近况", "当下",
    "2025", "2026", "2027",
    "新闻", "宣布", "新发布", "刚上线", "刚上市",
)

BOCHA_URL = "https://api.bochaai.com/v1/web-search"
SEARCH_COUNT = 8
SEARCH_TIMEOUT_S = 15.0


def normalize_freshness(fr: str | None) -> str:
    if not fr:
        return "noLimit"
    if fr in _FRESHNESS_ENUMS or _DATE_RE.match(fr):
        return fr
    return "noLimit"


def maybe_override_to_today(fr: str, user_text: str) -> str:
    if fr in ("oneDay", "noLimit") and any(kw in user_text for kw in TODAY_STRICT_KEYWORDS):
        return time.strftime("%Y-%m-%d")
    return fr


def _format_results(payload: dict, offset: int) -> tuple[str, list[dict]]:
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
        citations.append({"index": idx, "title": title, "url": url, "site": site, "date": date})
        parts.append(f"[{idx}] {title}\n来源: {site}{(' · ' + date) if date else ''}\n{summary}\nURL: {url}")
    return ("\n\n".join(parts) if parts else "(无搜索结果)"), citations


async def _bocha(query: str, freshness: str, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S) as c:
        r = await c.post(
            BOCHA_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query, "freshness": freshness, "summary": True, "count": SEARCH_COUNT},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from Bocha: {r.text[:400]}")
        return r.json()


def make_web_search_tool(get_api_key) -> Tool:
    """get_api_key: 返回当前 BOCHA_API_KEY 的可调用对象(延迟读取,便于热更新/测试)。"""

    async def handler(args: dict, ctx: ToolContext) -> ToolResult:
        key = get_api_key()
        if not key:
            return ToolResult(content="搜索失败: BOCHA_API_KEY 未设置")
        q = args.get("query", "") or ""
        fr = maybe_override_to_today(normalize_freshness(args.get("freshness", "noLimit") or "noLimit"), ctx.user_text)
        try:
            payload = await _bocha(q, fr, key)
            text, cits = _format_results(payload, offset=len(ctx.citations))
            ctx.citations.extend(cits)
            return ToolResult(content=text, state={"last_search_query": q})
        except Exception as e:
            return ToolResult(content=f"搜索失败: {e}")

    def on_start(args: dict) -> dict:
        return {"stage": "searching", "query": args.get("query", ""), "freshness": args.get("freshness", "")}

    return Tool(
        name="web_search",
        description=(
            "通过博查搜索引擎从主流公开网站(新闻、财经、百科、论文、官方站点)检索当前可获取的公开信息。"
            "这是检索公开数据,不是给投资/医疗/法律建议,你只是把结果转述给用户。\n\n"
            "必须调用(不允许以'我无法访问实时数据'等理由拒答):\n"
            "- 用户明确要求'搜索/联网/查一下/查询/帮我查'\n"
            "- 实时/近实时数据:股价、市值、汇率、商品价格、天气、赛事比分、航班/列车状态\n"
            "- 训练截止之后的事件:2025 年后的新闻、发布、人事、政策、论文、产品\n"
            "- '今天/最新/最近/本周/本月/刚刚/此刻'等时间限定的事实性问题\n"
            "- 你不熟悉的具体公司/产品/人物/论文,需核实事实\n\n"
            "不应调用:纯写作、代码/数学、静态常识、主观偏好。\n\n"
            "调用纪律:一次提问最多调用 1-2 次;不要并行发多个相似查询,不要按类别逐一检索;"
            "用一个综合且具体的查询词;结果不足时换关键词重搜一次为止;拿到结果直接作答。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "简洁的搜索关键词,接近搜索引擎写法,不要带'请帮我'之类口语"},
                "freshness": {
                    "type": "string",
                    "description": (
                        "时间范围过滤。两类取值:\n"
                        "(a) 枚举 noLimit/oneDay/oneWeek/oneMonth/oneYear\n"
                        "(b) 精确日期 YYYY-MM-DD\n"
                        "'今天/此刻/现在/刚刚'→今日 YYYY-MM-DD(别用 oneDay);'本周'→oneWeek;"
                        "'本月'→oneMonth;'今年'→oneYear;不确定→noLimit"
                    ),
                },
            },
            "required": ["query"],
        },
        handler=handler,
        available=lambda: bool(get_api_key()),
        force_keywords=FORCE_SEARCH_KEYWORDS,
        on_start=on_start,
    )
