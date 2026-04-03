"""
chat_service.py —— 对话服务
============================
职责：RAG 检索 + LLM 调用 + 结果返回
"""

import logging
from typing import List, Dict, Tuple, Optional

from src.core.llm_client import LLMClient
from src.core.rag_engine import RAGEngine
from src.config import DEFAULT_PROVIDER, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class ChatService:
    """对话服务类：封装完整的对话业务流程"""

    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        self.rag_engine = rag_engine

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        use_rag: bool = True,
        provider: str = None,
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, List[str]]:
        """核心对话方法"""
        logger.info(f"[ChatService] 收到对话请求: {message[:50]}...")

        provider = provider or DEFAULT_PROVIDER
        model = model or DEFAULT_MODEL
        client = LLMClient(provider=provider, model=model)

        rag_context = ""
        sources = []
        if use_rag and self.rag_engine and self.rag_engine.is_ready():
            try:
                rag_context, sources = self.rag_engine.get_context_with_sources(message)
                logger.info(f"[ChatService] RAG 检索完成，找到 {len(sources)} 个来源")
            except Exception as e:
                logger.error(f"[ChatService] RAG 检索失败: {e}")

        try:
            answer = client.chat(
                user_message=message,
                history=history or [],
                rag_context=rag_context,
                system_prompt=system_prompt or "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"[ChatService] LLM 调用成功")
            return answer, sources
        except Exception as e:
            logger.error(f"[ChatService] LLM 调用失败: {e}", exc_info=True)
            raise

    def get_knowledge_base_status(self) -> Dict[str, any]:
        """获取知识库状态"""
        if not self.rag_engine:
            return {"ready": False, "sources": [], "message": "RAG 引擎未初始化"}

        is_ready = self.rag_engine.is_ready()
        sources = self.rag_engine.list_sources() if is_ready else []

        return {
            "ready": is_ready,
            "sources": sources,
            "source_count": len(sources),
            "message": "知识库已就绪" if is_ready else "知识库未构建"
        }
