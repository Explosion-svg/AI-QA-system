"""
reranker.py —— 重排序模块
===========================
职责：用 cross-encoder 对候选文档重新打分，提升精准度。

Cross-Encoder vs Bi-Encoder：
  - Bi-Encoder（向量检索）：query 和 doc 分别编码，快但不够精准
  - Cross-Encoder（重排序）：query 和 doc 一起编码，慢但精准

工程级设计：
  - 只对 top-k 候选重排，控制计算量
  - 模型延迟加载，首次使用才下载
  - 支持批量推理，提升吞吐
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# 默认重排序模型（轻量级，中英文通用）
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    重排序器，使用 CrossEncoder 对候选文档精准打分。

    使用示例：
        reranker = Reranker()
        reranked = reranker.rerank("RAG 原理", candidates, top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        """延迟加载模型（首次调用时下载）。"""
        if self._model is None:
            logger.info("[Reranker] 正在加载模型: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("[Reranker] 模型加载完成")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        重排序候选文档。

        :param query: 查询文本
        :param candidates: 候选文档 [(doc, score), ...]
        :param top_k: 返回前 k 个
        :return: 重排序后的结果 [(doc, rerank_score), ...]
        """
        if not candidates:
            return []

        model = self._get_model()
        docs = [doc for doc, _ in candidates]
        pairs = [(query, doc.page_content) for doc in docs]

        # 批量推理
        scores = model.predict(pairs)

        # 按新分数排序
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        logger.debug("[Reranker] 重排序完成，输入 %d，输出 %d", len(candidates), len(reranked))
        return reranked
