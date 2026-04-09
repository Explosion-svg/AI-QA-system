"""
rag_engine.py —— RAG引擎
=========================
职责：RAG流程编排（rewrite → retrieve → rerank → filter → generate）
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.index.vector_store import VectorStore
from src.index.document_loader import DocumentLoader
from src.rag.rewrite import QueryRewriter
from src.rag.retriever import Retriever
from src.rag.rerank import Reranker
from src.rag.context_filter import ContextFilter

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG引擎
    负责文档索引构建和检索流程编排
    """

    def __init__(
        self,
        vector_store: VectorStore,
        document_loader: DocumentLoader,
        top_k: int = 4
    ):
        """
        初始化RAG引擎

        Args:
            vector_store: 向量存储
            document_loader: 文档加载器
            top_k: 检索top-k
        """
        self.vector_store = vector_store
        self.document_loader = document_loader
        self.top_k = top_k

        # RAG组件
        self.query_rewriter = QueryRewriter(enable_keyword_extract=True)
        self.retriever = Retriever(vector_store=vector_store)
        self.reranker = Reranker()
        self.context_filter = ContextFilter(min_length=50, max_total_length=4000)

        self._loaded = False

        logger.info("[RAGEngine] 初始化完成")

    def build_index(self, file_paths: List[str]) -> int:
        """
        构建索引

        Args:
            file_paths: 文件路径列表

        Returns:
            总chunk数
        """
        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.document_loader.load_and_split(file_path)
                all_chunks.extend(chunks)
                logger.info(f"[RAGEngine] 已加载: {Path(file_path).name}, {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"[RAGEngine] 加载失败 {file_path}: {e}")

        if all_chunks:
            # 添加到向量库
            self.vector_store.add_documents(all_chunks)
            self._loaded = True
            logger.info(f"[RAGEngine] 索引构建完成: {len(all_chunks)} chunks")

        return len(all_chunks)

    def load_index(self) -> bool:
        """
        加载已有索引

        Returns:
            是否加载成功
        """
        try:
            # 检查向量库
            if self.vector_store.is_empty():
                logger.warning("[RAGEngine] 向量库为空")
                return False

            self._loaded = True
            logger.info("[RAGEngine] 索引加载成功")
            return True

        except Exception as e:
            logger.error(f"[RAGEngine] 索引加载失败: {e}")
            return False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回top-k结果

        Returns:
            文档列表
        """
        if not self._loaded:
            logger.warning("[RAGEngine] 索引未加载")
            return []

        k = top_k or self.top_k

        try:
            # 1. 查询改写
            rewritten_queries = self.query_rewriter.rewrite(query)
            logger.info(f"[RAGEngine] 查询改写: {len(rewritten_queries)} 个变体")

            # 2. 向量检索（使用第一个改写查询）
            search_query = rewritten_queries[0] if rewritten_queries else query
            results = self.retriever.search(search_query, top_k=k * 2)

            # 3. 重排序
            if results:
                results = self.reranker.rerank(query, results, top_k=k)

            # 4. 上下文过滤
            if results:
                results = self.context_filter.filter(results)

            logger.info(f"[RAGEngine] 检索完成: {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"[RAGEngine] 检索失败: {e}")
            return []

    def get_context_with_sources(self, query: str) -> Tuple[str, List[str]]:
        """
        获取上下文和来源

        Args:
            query: 查询文本

        Returns:
            (上下文, 来源列表)
        """
        documents = self.retrieve(query)

        if not documents:
            return "", []

        # 提取上下文
        context = "\n\n".join([doc.page_content for doc in documents])

        # 提取来源
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "未知来源")
            if source not in sources:
                sources.append(source)

        return context, sources

    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self._loaded and not self.vector_store.is_empty()

    def list_sources(self) -> List[str]:
        """列出所有来源"""
        if not self._loaded:
            return []

        try:
            # 从向量库获取所有唯一来源
            # 这里简化处理，实际可以查询向量库的metadata
            return []
        except Exception as e:
            logger.error(f"[RAGEngine] 获取来源失败: {e}")
            return []

    def clear_index(self) -> None:
        """清空索引"""
        try:
            self.vector_store.delete_collection()
            self.retriever.clear_index()
            self._loaded = False
            logger.info("[RAGEngine] 索引已清空")
        except Exception as e:
            logger.error(f"[RAGEngine] 清空索引失败: {e}")
            raise

    def close(self) -> None:
        """关闭引擎"""
        self.vector_store.close()
        self._loaded = False
        logger.info("[RAGEngine] 引擎已关闭")
