"""
retriever.py —— 向量检索器
===========================
职责：基于向量相似度的语义检索
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from src.index.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    向量检索器
    基于语义相似度进行文档检索
    """

    def __init__(self, vector_store: VectorStore):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        logger.info("[Retriever] 初始化完成")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Document]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果

        Returns:
            文档列表
        """
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"[Retriever] 检索完成: {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"[Retriever] 检索失败: {e}")
            return []

    def search_with_score(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        带分数的向量检索

        Args:
            query: 查询文本
            top_k: 返回top-k结果

        Returns:
            (文档, 分数)列表
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            logger.info(f"[Retriever] 检索完成: {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"[Retriever] 检索失败: {e}")
            return []
