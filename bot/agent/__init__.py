"""Agent 引擎包:可插拔工具 + 模型驱动的工具循环 + 可替换上游 + 上下文管理。"""
from .engine import Agent, AgentConfig
from .context import ContextConfig, ContextManager, estimate_tokens, total_tokens
from .tools import Tool, ToolContext, ToolRegistry, ToolResult
from . import builtins

__all__ = [
    "Agent", "AgentConfig",
    "ContextConfig", "ContextManager", "estimate_tokens", "total_tokens",
    "Tool", "ToolContext", "ToolRegistry", "ToolResult",
    "builtins",
]
