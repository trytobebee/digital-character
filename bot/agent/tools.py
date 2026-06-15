"""
工具抽象层 —— Agent 引擎与具体能力解耦的边界。

设计目标:加一个工具 = 写一个 handler + 注册,**不碰引擎主循环**。

核心三件套:
  Tool         一个工具的完整描述(元数据 + handler)
  ToolResult   handler 的返回:给模型读的 content + 推给前端的 events
  ToolContext  一次请求内所有工具共享的便笺(指代消解 / 跨工具传值 / 引用累加)
  ToolRegistry 工具登记处,按"是否可用 / 关键词硬路由"筛选
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolResult:
    """一次工具执行的产物。

    content : 回写给模型的 tool 消息正文(模型据此继续推理)。
    events  : 额外推给前端 SSE 的事件(如 citations);引擎会逐条 yield。
    state   : 要并入 ToolContext.state 的键值(供后续工具/轮次读取,如指代消解)。
    """
    content: str
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """一次请求/一轮 CLI 任务内,所有工具共享的运行期上下文。"""
    user_text: str                       # 最后一条用户消息的纯文本(决策/硬路由/freshness 用)
    state: dict[str, Any] = field(default_factory=dict)   # 跨工具便笺(实体、上一步结果……)
    citations: list[dict[str, Any]] = field(default_factory=list)  # 全局引用累加器
    workdir: str = "."                   # 工具的工作根目录(CLI coding agent 用)
    # 危险操作(写文件/执行命令)的确认回调,返回 True 才放行。
    # 签名: async (action: str, preview: str | None) -> bool。None 表示无人值守自动放行。
    confirm: Any = None


# handler 签名:(已解析的参数 dict, 上下文) -> ToolResult
ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema(OpenAI function.parameters)
    handler: ToolHandler
    # 是否在本次请求挂给模型(web_search 需 key,get_current_time 永远在)
    available: Callable[[], bool] = field(default=lambda: True)
    # 命中任一关键词 → 第 1 轮强制 tool_choice 指向本工具,绕过弱模型的拒答反射/历史污染
    force_keywords: tuple[str, ...] = ()
    # 执行前推给前端的进度事件(如 web_search 显示"搜索中: <query>");返回 None 用引擎默认
    on_start: Callable[[dict[str, Any]], dict[str, Any] | None] = field(default=lambda args: None)

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI tools 数组里的一项。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具登记处。server 在装配请求时用它筛出可用工具 + 判断硬路由目标。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def available_tools(self) -> list[Tool]:
        """本次请求实际可用(available() 为真)的工具列表。"""
        return [t for t in self._tools.values() if t.available()]

    def forced_tool(self, tools: list[Tool], user_text: str) -> Tool | None:
        """在给定工具集中,找第一个 force_keywords 命中用户文本的工具(硬路由目标)。"""
        for t in tools:
            if t.force_keywords and any(kw in user_text for kw in t.force_keywords):
                return t
        return None
