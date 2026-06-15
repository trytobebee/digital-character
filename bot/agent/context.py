"""
上下文管理(Tier 1:确定性,不依赖模型)。

借鉴 Claude Code 的做法,但只抄"不靠模型摘要"的那几招——对弱模型最安全:
  · 工具结果折叠   旧的 read_file/grep/web_search 结果最占地、最易过时,
                   超出"最近 N 条"且体积大的,正文换成短存根(协议字段保留)
  · token 预算     估算总 token,只在越过软目标时才折叠(待在 prefill 快的甜点区)
  · 文件读取追踪   配合 ctx.state,edit/write 后让旧的 read 结果作废(由工具侧调用)

不折叠:system、最近一轮、assistant 的思考/答复正文(保留推理主线)。
折叠是幂等的:存根不会被二次折叠。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")
_FOLD_MARK = "⟨folded⟩"


def estimate_tokens(text: str) -> int:
    """粗略 token 估算(无需加载分词器):CJK 约 1 字 1 token,其余约 4 字 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "image_url":
                parts.append(" " * 4096)  # 图像按 ~1024 token 计入预算(占位)
        return "".join(parts)
    return ""


def message_tokens(msg: dict) -> int:
    t = estimate_tokens(_content_text(msg.get("content")))
    # 工具调用参数也占 token
    for tc in msg.get("tool_calls") or []:
        t += estimate_tokens(((tc.get("function") or {}).get("arguments")) or "")
    return t + 4  # 每条消息的角色/分隔开销


def total_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


@dataclass
class ContextConfig:
    soft_budget: int = 24_000        # 越过此值才开始折叠(留在 prefill 快的区间)
    keep_recent_tool_results: int = 4  # 最近 N 条工具结果永远保留原文
    min_fold_chars: int = 400        # 小于此长度的工具结果不值得折叠


class ContextManager:
    def __init__(self, config: ContextConfig | None = None) -> None:
        self.cfg = config or ContextConfig()

    @staticmethod
    def _tool_name_for(messages: list[dict], tool_call_id: str) -> str:
        """用 tool_call_id 反查前面 assistant.tool_calls 里的工具名(不污染发往 API 的 payload)。"""
        for m in messages:
            for tc in m.get("tool_calls") or []:
                if tc.get("id") == tool_call_id:
                    return ((tc.get("function") or {}).get("name")) or "工具"
        return "工具"

    def manage(self, messages: list[dict]) -> dict:
        """就地整理 messages。返回本次动作的统计 dict(供引擎吐事件)。"""
        before = total_tokens(messages)
        if before <= self.cfg.soft_budget:
            return {"folded": 0, "before": before, "after": before, "saved": 0}

        # 找出所有"可折叠"的工具消息下标:role==tool、未折叠、体积够大
        tool_idxs = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool"
            and _FOLD_MARK not in (m.get("content") or "")
            and len(m.get("content") or "") >= self.cfg.min_fold_chars
        ]
        # 保留最近 N 条工具结果原文,只折叠更老的(从最老开始)
        foldable = tool_idxs[: max(0, len(tool_idxs) - self.cfg.keep_recent_tool_results)]

        folded = 0
        for i in foldable:
            if total_tokens(messages) <= self.cfg.soft_budget:
                break
            m = messages[i]
            name = self._tool_name_for(messages, m.get("tool_call_id", ""))
            orig_len = len(m.get("content") or "")
            m["content"] = (
                f"{_FOLD_MARK} 此处是较早的{name}结果(约 {orig_len} 字),"
                "为节省上下文已折叠。如仍需要,请重新调用相应工具获取最新内容。"
            )
            folded += 1

        after = total_tokens(messages)
        return {"folded": folded, "before": before, "after": after, "saved": before - after}
