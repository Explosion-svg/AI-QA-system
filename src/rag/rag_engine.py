"""
rag_engine.py —— RAG 流程编排
==============================
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

from src.index.document_loader import DocumentLoader
from src.index.vector_store import VectorStore
from src.rag.context_filter import ContextFilter
from src.rag.rerank import Reranker
from src.rag.retriever import Retriever
from src.rag.rewrite import QueryRewriter
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGEngine:
    """负责索引构建、启动加载和查询检索全流程。"""

    def __init__(
        self,
        vector_store: VectorStore,
        document_loader: DocumentLoader,
        dense_top_k: int,
        sparse_top_k: int,
        fusion_top_k: int,
        rerank_top_k: int,
        context_top_k: int,
        rrf_k: int,
        min_chunk_length: int,
        max_context_length: int,
    ):
        self.vector_store = vector_store
        self.document_loader = document_loader
        self.query_rewriter = QueryRewriter(enable_keyword_extract=True)
        self.retriever = Retriever(
            vector_store=vector_store,
            query_rewriter=self.query_rewriter,
            rrf_k=rrf_k,
        )
        self.reranker = Reranker()
        self.context_filter = ContextFilter(
            min_length=min_chunk_length,
            max_total_length=max_context_length,
        )

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.rerank_top_k = rerank_top_k
        self.context_top_k = context_top_k
        self._loaded = False

        logger.info("[RAGEngine] 初始化完成")

    def build_index(self, file_paths: List[str]) -> int:
        all_chunks = []
        indexed_sources = []
        start = perf_counter()

        for file_path in file_paths:
            try:
                chunks = self.document_loader.load_and_split(file_path)
                if not chunks:
                    continue
                source = Path(file_path).name
                indexed_sources.append(source)
                all_chunks.extend(chunks)
                logger.info(
                    "[RAGEngine] 文件切块完成 source=%s chunks=%d",
                    source,
                    len(chunks),
                )
            except Exception as exc:
                logger.exception("[RAGEngine] 加载失败 file=%s error=%s", file_path, exc)

        if not all_chunks:
            return 0

        self.vector_store.delete_by_source(indexed_sources)
        chunk_count = self.vector_store.upsert_documents(all_chunks)
        self._refresh_retriever()
        self._loaded = chunk_count > 0

        logger.info(
            "[RAGEngine] 索引构建完成 files=%d chunks=%d cost=%.3fs",
            len(indexed_sources),
            chunk_count,
            perf_counter() - start,
        )
        return chunk_count

    def load_index(self) -> bool:
        try:
            if self.vector_store.is_empty():
                logger.warning("[RAGEngine] 向量库为空")
                self.retriever.clear_index()
                self._loaded = False
                return False

            self._refresh_retriever()
            self._loaded = True
            logger.info(
                "[RAGEngine] 索引加载成功 chunks=%d sources=%d",
                self.vector_store.get_collection_count(),
                len(self.vector_store.list_sources()),
            )
            return True
        except Exception as exc:
            logger.exception("[RAGEngine] 索引加载失败: %s", exc)
            self._loaded = False
            return False

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if not self._loaded:
            logger.warning("[RAGEngine] 索引未加载")
            return []

        start = perf_counter()
        rewritten_queries = self.query_rewriter.rewrite(query)
        candidates = self.retriever.search_hybrid(
            queries=rewritten_queries,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
            fusion_top_k=self.fusion_top_k,
        )

        if not candidates:
            logger.info("[RAGEngine] 未召回到候选 query=%s", query)
            return []

        reranked = self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        filtered = self.context_filter.filter(reranked, top_k=self.context_top_k)

        logger.info(
            "[RAGEngine] 检索完成 rewrite=%d candidates=%d final=%d cost=%.3fs",
            len(rewritten_queries),
            len(candidates),
            len(filtered),
            perf_counter() - start,
        )
        return filtered

    def get_context_with_sources(self, query: str) -> Tuple[str, List[str]]:
        chunks = self.retrieve(query)
        if not chunks:
            return "", []

        context_blocks = []
        sources: List[str] = []
        for chunk in chunks:
            page = chunk.metadata.get("page")
            page_suffix = f" | page={page}" if page is not None else ""
            header = f"[source={chunk.source}{page_suffix} | chunk={chunk.chunk_id[:8]}]"
            context_blocks.append(f"{header}\n{chunk.page_content}")
            if chunk.source not in sources:
                sources.append(chunk.source)

        return "\n\n".join(context_blocks), sources

    def is_ready(self) -> bool:
        return self._loaded and not self.vector_store.is_empty()

    def list_sources(self) -> List[str]:
        return self.vector_store.list_sources() if self._loaded else []

    def clear_index(self) -> None:
        self.vector_store.delete_collection()
        self.retriever.clear_index()
        self._loaded = False
        logger.info("[RAGEngine] 索引已清空")

    def close(self) -> None:
        self.retriever.clear_index()
        self.vector_store.close()
        self._loaded = False
        logger.info("[RAGEngine] 引擎已关闭")

    def _refresh_retriever(self) -> None:
        documents = self.vector_store.get_all_documents()
        self.retriever.refresh_index(documents)
