"""
retriever.py —— 混合检索器
===========================
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, List, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.index.vector_store import VectorStore
from src.rag.rewrite import QueryRewriter
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """负责 dense + sparse 混合召回及 RRF 融合。"""

    def __init__(self, vector_store: VectorStore, query_rewriter: QueryRewriter, rrf_k: int = 60):
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter
        self.rrf_k = rrf_k
        self._documents: List[Document] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: List[List[str]] = []

        logger.info("[Retriever] 初始化完成")

    def refresh_index(self, documents: Sequence[Document]) -> None:
        """
        BM25 稀疏检索引擎的「重建索引」
        :param documents:
        :return:
        """
        self._documents = list(documents)
        self._corpus_tokens = [
            self.query_rewriter.tokenize(document.page_content) or ["__empty__"]
            for document in self._documents
        ]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("[Retriever] 稀疏索引已刷新，chunks=%d", len(self._documents))

    def clear_index(self) -> None:
        self._documents = []
        self._corpus_tokens = []
        self._bm25 = None

    def search_hybrid(
        self,
        queries: Sequence[str],
        dense_top_k: int,
        sparse_top_k: int,
        fusion_top_k: int,
    ) -> List[RetrievedChunk]:
        start = perf_counter()
        dense_results = self._dense_recall(queries, dense_top_k)
        sparse_results = self._sparse_recall(queries, sparse_top_k)
        fused_results = self._rrf_fuse(dense_results, sparse_results, fusion_top_k)

        logger.info(
            "[Retriever] hybrid 检索完成 dense=%d sparse=%d fused=%d cost=%.3fs",
            len(dense_results),
            len(sparse_results),
            len(fused_results),
            perf_counter() - start,
        )
        return fused_results

    def _dense_recall(self, queries: Sequence[str], top_k: int) -> List[RetrievedChunk]:
        """
        向量检索(语义匹配)
        :param queries:
        :param top_k:
        :return:
        """
        merged: Dict[str, RetrievedChunk] = {}
        rank_offset = 0

        for query in queries:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            for rank, (document, distance) in enumerate(results, start=1):
                chunk_id = document.metadata.get("chunk_id")
                if not chunk_id:
                    continue
                item = merged.get(chunk_id)
                dense_score = 1.0 / (1.0 + max(distance, 0.0))
                if item is None or (item.dense_rank or 10 ** 9) > rank + rank_offset:
                    merged[chunk_id] = RetrievedChunk(
                        document=document,
                        chunk_id=chunk_id,
                        source=document.metadata.get("source", "未知来源"),
                        dense_rank=rank + rank_offset,
                        dense_score=dense_score,
                    )
            rank_offset += top_k

        return sorted(
            merged.values(),
            key=lambda item: (item.dense_rank or 10 ** 9, -(item.dense_score or 0.0)),
        )

    def _sparse_recall(self, queries: Sequence[str], top_k: int) -> List[RetrievedChunk]:
        """
        稀疏检索（BM25 关键词精确匹配）
        :param queries:
        :param top_k:
        :return:
        """
        if not self._bm25 or not self._documents:
            return []

        merged: Dict[str, RetrievedChunk] = {}
        rank_offset = 0
        for query in queries:
            tokens = self.query_rewriter.tokenize(query)
            if not tokens:
                continue

            scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]

            for rank, (doc_index, score) in enumerate(ranked, start=1):
                document = self._documents[doc_index]
                chunk_id = document.metadata.get("chunk_id")
                if not chunk_id:
                    continue
                item = merged.get(chunk_id)
                if item is None or (item.sparse_rank or 10 ** 9) > rank + rank_offset:
                    merged[chunk_id] = RetrievedChunk(
                        document=document,
                        chunk_id=chunk_id,
                        source=document.metadata.get("source", "未知来源"),
                        sparse_rank=rank + rank_offset,
                        sparse_score=float(score),
                    )
            rank_offset += top_k

        return sorted(
            merged.values(),
            key=lambda item: (item.sparse_rank or 10 ** 9, -(item.sparse_score or 0.0)),
        )

    def _rrf_fuse(
        self,
        dense_results: Sequence[RetrievedChunk],
        sparse_results: Sequence[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
         分数融合（把两路结果合并排序）
        :param dense_results:
        :param sparse_results:
        :param top_k:
        :return:
        """
        merged: Dict[str, RetrievedChunk] = {}

        for candidate in dense_results:
            merged[candidate.chunk_id] = candidate

        for candidate in sparse_results:
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
                continue
            existing.sparse_rank = candidate.sparse_rank
            existing.sparse_score = candidate.sparse_score

        for candidate in merged.values():
            fusion_score = 0.0
            if candidate.dense_rank is not None:
                fusion_score += 1.0 / (self.rrf_k + candidate.dense_rank)
            if candidate.sparse_rank is not None:
                fusion_score += 1.0 / (self.rrf_k + candidate.sparse_rank)
            candidate.fusion_score = fusion_score

        return sorted(
            merged.values(),
            key=lambda item: (
                -item.fusion_score,
                item.dense_rank or 10 ** 9,
                item.sparse_rank or 10 ** 9,
            ),
        )[:top_k]
