"""
rerank.py —— 候选重排序
========================
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import List

from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """使用 CrossEncoder 对候选文档精准重排。"""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("[Reranker] 加载模型: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        start = perf_counter()
        try:
            model = self._get_model()
            pairs = [(query, item.page_content) for item in candidates]
            scores = model.predict(pairs)

            reranked: List[RetrievedChunk] = []
            for item, score in zip(candidates, scores):
                item.rerank_score = float(score)
                reranked.append(item)

            reranked.sort(
                key=lambda item: (
                    -(item.rerank_score or 0.0),
                    -item.fusion_score,
                )
            )

            logger.info(
                "[Reranker] 重排完成 input=%d output=%d cost=%.3fs",
                len(candidates),
                min(len(reranked), top_k),
                perf_counter() - start,
            )
            return reranked[:top_k]
        except Exception as exc:
            logger.exception("[Reranker] 重排失败，降级为 fusion 排序: %s", exc)
            return candidates[:top_k]
