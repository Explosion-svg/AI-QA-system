"""
chat_service.py —— 对话业务服务
================================
职责：编排聊天业务流程（加载历史、调用RAG、保存历史、返回回答）
"""

import logging
from time import perf_counter
from typing import List, Tuple, Optional

from src.rag.rag_engine import RAGEngine
from src.memory.history_manager import HistoryManager
from src.memory.memory_manager import MemoryManager
from src.infra.llm_client import LLMClient
from src.infra.config import DEFAULT_MODEL, DEFAULT_PROVIDER

logger = logging.getLogger(__name__)


class ChatService:
    """
    聊天业务服务
    负责编排完整的对话流程
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        history_manager: HistoryManager,
        memory_manager: MemoryManager,
    ):
        """
        初始化聊天服务

        Args:
            rag_engine: RAG引擎
            history_manager: 历史管理器
        """
        self.rag_engine = rag_engine
        self.history_manager = history_manager
        self.memory_manager = memory_manager

        logger.info("[ChatService] 初始化完成")

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_rag: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Tuple[str, List[str]]:
        """
        处理聊天请求

        Args:
            message: 用户消息
            session_id: 会话ID
            use_rag: 是否使用RAG
            provider: LLM提供商
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            (回答, 来源列表)
        """
        total_start = perf_counter()
        logger.info("[ChatService] 处理聊天 message=%s", message[:80])

        # 1. 加载历史
        history_start = perf_counter()
        # 新增：优先从结构化记忆中构造上下文，不再依赖简单尾部裁剪。
        history = self.memory_manager.build_context_messages(session_id)
        logger.info(
            "[ChatService] 历史加载完成 session_id=%s messages=%d cost=%.3fs",
            session_id,
            len(history),
            perf_counter() - history_start,
        )

        # 2. RAG检索
        rag_start = perf_counter()
        rag_context = ""
        sources = []
        if use_rag and self.rag_engine.is_ready():
            try:
                rag_context, sources = self.rag_engine.get_context_with_sources(message)
                logger.info(
                    "[ChatService] RAG检索完成 sources=%d context_chars=%d cost=%.3fs",
                    len(sources),
                    len(rag_context),
                    perf_counter() - rag_start,
                )
            except Exception as e:
                logger.error(f"[ChatService] RAG检索失败: {e}")
        else:
            logger.info(
                "[ChatService] 跳过RAG use_rag=%s rag_ready=%s",
                use_rag,
                self.rag_engine.is_ready(),
            )

        llm_client = LLMClient(
            provider=provider or DEFAULT_PROVIDER,
            model=model or DEFAULT_MODEL,
        )

        # 4. 调用LLM
        try:
            llm_start = perf_counter()
            answer = llm_client.chat(
                user_message=message,
                history=history,
                rag_context=rag_context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(
                "[ChatService] LLM调用成功 provider=%s model=%s cost=%.3fs",
                llm_client.provider,
                llm_client.model,
                perf_counter() - llm_start,
            )
        except Exception as e:
            logger.error(f"[ChatService] LLM调用失败: {e}", exc_info=True)
            raise

        # 5. 保存历史
        if session_id:
            # 新增：由 MemoryManager 统一处理“追加 + 压缩 + 持久化”。
            self.memory_manager.append_turn(
                session_id=session_id,
                user=message,
                assistant=answer,
                provider=llm_client.provider,
                model=llm_client.model,
                meta={"provider": llm_client.provider, "model": llm_client.model},
            )
            logger.info("[ChatService] 历史已保存")

        logger.info("[ChatService] 请求完成 total_cost=%.3fs", perf_counter() - total_start)
        return answer, sources

    def get_knowledge_base_status(self) -> dict:
        """
        获取知识库状态

        Returns:
            状态信息
        """
        if not self.rag_engine:
            return {
                "ready": False,
                "sources": [],
                "source_count": 0,
                "message": "RAG引擎未初始化"
            }

        is_ready = self.rag_engine.is_ready()
        sources = self.rag_engine.list_sources() if is_ready else []

        return {
            "ready": is_ready,
            "sources": sources,
            "source_count": len(sources),
            "message": "知识库已就绪" if is_ready else "知识库未构建"
        }

    def get_history(self, session_id: str) -> List[dict]:
        """
        获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            历史记录
        """
        return self.history_manager.load(session_id)

    def clear_history(self, session_id: str) -> None:
        """
        清空会话历史

        Args:
            session_id: 会话ID
        """
        self.history_manager.delete(session_id)
        logger.info(f"[ChatService] 已清空会话历史: {session_id}")
