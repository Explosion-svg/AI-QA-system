"""
chat_service.py —— 对话业务服务
================================
职责：编排聊天业务流程（加载历史、调用RAG、保存历史、返回回答）
"""

import logging
from typing import List, Tuple, Optional

from src.rag.rag_engine import RAGEngine
from src.memory.history_manager import HistoryManager
from src.infra.llm_client import LLMClient

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
        llm_client: LLMClient
    ):
        """
        初始化聊天服务

        Args:
            rag_engine: RAG引擎
            history_manager: 历史管理器
            llm_client: LLM客户端
        """
        self.rag_engine = rag_engine
        self.history_manager = history_manager
        self.llm_client = llm_client

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
        logger.info(f"[ChatService] 处理聊天: {message[:50]}...")

        # 1. 加载历史
        history = []
        if session_id:
            history = self.history_manager.load(session_id)
            logger.info(f"[ChatService] 加载历史: {len(history)} 条")

        # 2. RAG检索
        rag_context = ""
        sources = []
        if use_rag and self.rag_engine.is_ready():
            try:
                rag_context, sources = self.rag_engine.get_context_with_sources(message)
                logger.info(f"[ChatService] RAG检索完成: {len(sources)} 个来源")
            except Exception as e:
                logger.error(f"[ChatService] RAG检索失败: {e}")

        # 3. 切换模型（如果指定）
        if provider and model:
            self.llm_client.switch(provider, model)

        # 4. 调用LLM
        try:
            answer = self.llm_client.chat(
                user_message=message,
                history=history,
                rag_context=rag_context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info("[ChatService] LLM调用成功")
        except Exception as e:
            logger.error(f"[ChatService] LLM调用失败: {e}", exc_info=True)
            raise

        # 5. 保存历史
        if session_id:
            # 加载现有历史
            existing_history = self.history_manager.load(session_id)
            # 添加新的对话
            existing_history.append({"role": "user", "content": message})
            existing_history.append({"role": "assistant", "content": answer})
            # 保存更新后的历史
            self.history_manager.save(session_id, existing_history)
            logger.info("[ChatService] 历史已保存")

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
