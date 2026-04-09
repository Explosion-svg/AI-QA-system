"""
memory_manager.py —— 会话记忆管理器
====================================
职责：
1. 管理“摘要 + 最近消息”的分层记忆结构
2. 在上下文过长时调用 LLM 压缩前半段消息
3. 对外提供可直接喂给 LLM 的消息上下文
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.infra.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    MEMORY_RECENT_MAX_CHARS,
    MEMORY_ROLLING_MAX_CHARS,
    MEMORY_SUMMARY_MAX_CHARS,
)
from src.infra.llm_client import LLMClient
from src.memory.history_manager import HistoryManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    新增：真正负责“会话记忆压缩”的管理器。
    """

    def __init__(
        self,
        history_manager: HistoryManager,
        recent_max_chars: int = MEMORY_RECENT_MAX_CHARS,
        rolling_max_chars: int = MEMORY_ROLLING_MAX_CHARS,
        summary_max_chars: int = MEMORY_SUMMARY_MAX_CHARS,
    ):
        self.history_manager = history_manager
        self.recent_max_chars = recent_max_chars
        self.rolling_max_chars = rolling_max_chars
        self.summary_max_chars = summary_max_chars

    def build_context_messages(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id:
            return []
        memory = self.history_manager.load_memory_state(session_id)
        return self.history_manager.build_messages_from_memory(memory)

    def append_turn(
        self,
        session_id: str,
        user: str,
        assistant: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        新增：追加一轮对话，并在必要时自动压缩记忆。
        """
        memory = self.history_manager.load_memory_state(session_id)
        memory["recent_messages"].append({"role": "user", "content": user})
        memory["recent_messages"].append({"role": "assistant", "content": assistant})
        memory["stats"]["total_turns"] = int(memory["stats"].get("total_turns", 0)) + 1

        client = LLMClient(
            provider=provider or DEFAULT_PROVIDER,
            model=model or DEFAULT_MODEL,
        )

        try:
            self._compress_recent_if_needed(memory, client)
            self._compress_summary_if_needed(memory, client)
        except Exception as e:
            logger.warning("[MemoryManager] 记忆压缩失败，回退到有限窗口模式: %s", e)
            max_messages = self.history_manager.max_history * 2
            memory["recent_messages"] = memory["recent_messages"][-max_messages:]

        self.history_manager.save_memory_state(session_id, memory, meta=meta)
        return memory

    def _compress_recent_if_needed(self, memory: Dict[str, Any], client: LLMClient) -> None:
        recent_messages = memory.get("recent_messages", [])
        if len(recent_messages) < 4:
            return
        if self._messages_chars(recent_messages) <= self.recent_max_chars:
            return

        # 新增：窗口过长时，不再直接删除最老消息，而是压缩前半段。
        split_index = max(2, (len(recent_messages) // 2))
        if split_index % 2 != 0:
            split_index -= 1

        earlier = recent_messages[:split_index]
        later = recent_messages[split_index:]
        summary = self._summarize_messages(
            earlier,
            client,
            title="近期对话压缩摘要",
            instruction=(
                "请把以下多轮对话压缩为中文摘要。"
                "保留：用户目标、已经确认的事实、关键约束、未完成事项、重要结论。"
                "不要编造，不要遗漏用户明确提出的偏好。"
            ),
        )

        memory["rolling_summary"] = self._merge_text(memory.get("rolling_summary", ""), summary)
        memory["recent_messages"] = later
        memory["stats"]["summary_version"] = int(memory["stats"].get("summary_version", 0)) + 1

    def _compress_summary_if_needed(self, memory: Dict[str, Any], client: LLMClient) -> None:
        rolling_summary = (memory.get("rolling_summary") or "").strip()
        if len(rolling_summary) <= self.rolling_max_chars:
            return

        merged = self._merge_text(memory.get("session_summary", ""), rolling_summary)
        session_summary = self._summarize_text(
            merged,
            client,
            title="长期会话摘要",
            instruction=(
                "请把以下历史摘要提炼成更稳定的长期记忆。"
                "保留：用户长期目标、偏好、上下文事实、关键决策、遗留问题。"
                f"输出尽量控制在 {self.summary_max_chars} 个中文字符以内。"
            ),
        )
        memory["session_summary"] = session_summary
        memory["rolling_summary"] = ""

    def _summarize_messages(
        self,
        messages: List[Dict[str, str]],
        client: LLMClient,
        title: str,
        instruction: str,
    ) -> str:
        conversation = []
        for msg in messages:
            role = "用户" if msg.get("role") == "user" else "助手"
            conversation.append(f"{role}: {msg.get('content', '')}")
        return self._summarize_text("\n".join(conversation), client, title, instruction)

    def _summarize_text(
        self,
        text: str,
        client: LLMClient,
        title: str,
        instruction: str,
    ) -> str:
        prompt = f"{instruction}\n\n{text}"
        return client.chat(
            user_message=prompt,
            history=[],
            rag_context="",
            system_prompt=f"你是一个负责会话记忆压缩的助手，请只输出{title}本身。",
            temperature=0.2,
            max_tokens=1024,
        ).strip()

    @staticmethod
    def _messages_chars(messages: List[Dict[str, str]]) -> int:
        return sum(len(msg.get("content", "")) for msg in messages)

    @staticmethod
    def _merge_text(old_text: str, new_text: str) -> str:
        old_text = (old_text or "").strip()
        new_text = (new_text or "").strip()
        if old_text and new_text:
            return f"{old_text}\n\n{new_text}"
        return old_text or new_text
