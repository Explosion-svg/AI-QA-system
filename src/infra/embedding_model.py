"""
embedding_model.py —— 文本向量化模型
====================================
职责：封装embedding模型，提供文本向量化能力
"""

import logging
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding模型封装类
    负责将文本转换为向量表示
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化embedding模型

        Args:
            model_name: HuggingFace模型名称
        """
        self.model_name = model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        logger.info(f"[EmbeddingModel] 初始化: {model_name}")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        获取embedding模型实例（延迟加载）

        Returns:
            HuggingFaceEmbeddings实例
        """
        if self._embeddings is None:
            logger.info(f"[EmbeddingModel] 加载模型: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("[EmbeddingModel] 模型加载完成")
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            向量表示
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        embeddings = self.get_embeddings()
        return embeddings.embed_documents(texts)
