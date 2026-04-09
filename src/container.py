"""
container.py —— 依赖注入容器
==============================
职责：系统对象创建中心，管理所有核心对象的生命周期
"""

import logging
from typing import Optional

from src.infra.embedding_model import EmbeddingModel
from src.infra.config import (
    CHAT_SAVE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_HISTORY,
    RAG_CONTEXT_TOP_K,
    RAG_DENSE_TOP_K,
    RAG_FUSION_TOP_K,
    RAG_MAX_CONTEXT_LENGTH,
    RAG_MIN_CHUNK_LENGTH,
    RAG_RERANK_TOP_K,
    RAG_SPARSE_TOP_K,
    RRF_K,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
)
from src.index.chroma_store import ChromaStore
from src.index.document_loader import DocumentLoader
from src.rag.rag_engine import RAGEngine
from src.memory.history_manager import HistoryManager
from src.memory.memory_manager import MemoryManager
from src.services.chat_service import ChatService
from src.services.upload_service import UploadService

logger = logging.getLogger(__name__)


class Container:
    """
    依赖注入容器
    负责创建和管理所有核心对象
    """

    def __init__(self):
        """初始化容器"""
        self._embedding_model: Optional[EmbeddingModel] = None

        # 索引层
        self._vector_store: Optional[ChromaStore] = None
        self._document_loader: Optional[DocumentLoader] = None

        # RAG层
        self._rag_engine: Optional[RAGEngine] = None

        # 记忆层
        self._history_manager: Optional[HistoryManager] = None
        self._memory_manager: Optional[MemoryManager] = None

        # 业务层
        self._chat_service: Optional[ChatService] = None
        self._upload_service: Optional[UploadService] = None

        logger.info("[Container] 容器初始化完成")

    # ============================================================
    # 基础设施层
    # ============================================================

    def embedding_model(self) -> EmbeddingModel:
        """获取Embedding模型"""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL)
            logger.info("[Container] EmbeddingModel已创建")
        return self._embedding_model

    # ============================================================
    # 索引层
    # ============================================================

    def vector_store(self) -> ChromaStore:
        """获取向量存储"""
        if self._vector_store is None:
            self._vector_store = ChromaStore(
                collection_name=VECTOR_COLLECTION_NAME,
                persist_directory=VECTOR_DB_DIR,
                embedding_model=self.embedding_model()
            )
            logger.info("[Container] ChromaStore已创建")
        return self._vector_store

    def document_loader(self) -> DocumentLoader:
        """获取文档加载器"""
        if self._document_loader is None:
            self._document_loader = DocumentLoader(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            logger.info("[Container] DocumentLoader已创建")
        return self._document_loader

    # ============================================================
    # RAG层
    # ============================================================

    def rag_engine(self) -> RAGEngine:
        """获取RAG引擎"""
        if self._rag_engine is None:
            self._rag_engine = RAGEngine(
                vector_store=self.vector_store(),
                document_loader=self.document_loader(),
                dense_top_k=RAG_DENSE_TOP_K,
                sparse_top_k=RAG_SPARSE_TOP_K,
                fusion_top_k=RAG_FUSION_TOP_K,
                rerank_top_k=RAG_RERANK_TOP_K,
                context_top_k=RAG_CONTEXT_TOP_K,
                rrf_k=RRF_K,
                min_chunk_length=RAG_MIN_CHUNK_LENGTH,
                max_context_length=RAG_MAX_CONTEXT_LENGTH,
            )
            logger.info("[Container] RAGEngine已创建")
        return self._rag_engine

    # ============================================================
    # 记忆层
    # ============================================================

    def history_manager(self) -> HistoryManager:
        """获取历史管理器"""
        if self._history_manager is None:
            self._history_manager = HistoryManager(
                save_dir=CHAT_SAVE_DIR,
                max_history=MAX_HISTORY
            )
            logger.info("[Container] HistoryManager已创建")
        return self._history_manager

    def memory_manager(self) -> MemoryManager:
        """获取会话记忆管理器"""
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(
                history_manager=self.history_manager(),
            )
            logger.info("[Container] MemoryManager已创建")
        return self._memory_manager

    # ============================================================
    # 业务层
    # ============================================================

    def chat_service(self) -> ChatService:
        """获取聊天服务"""
        if self._chat_service is None:
            self._chat_service = ChatService(
                rag_engine=self.rag_engine(),
                history_manager=self.history_manager(),
                memory_manager=self.memory_manager(),
            )
            logger.info("[Container] ChatService已创建")
        return self._chat_service

    def upload_service(self) -> UploadService:
        """获取上传服务"""
        if self._upload_service is None:
            self._upload_service = UploadService(
                rag_engine=self.rag_engine(),
                document_loader=self.document_loader()
            )
            logger.info("[Container] UploadService已创建")
        return self._upload_service

    # ============================================================
    # 生命周期管理
    # ============================================================

    async def startup(self) -> None:
        """系统启动时调用"""
        logger.info("[Container] 系统启动中...")

        # 加载RAG索引
        rag_engine = self.rag_engine()
        if rag_engine.load_index():
            logger.info("[Container] RAG索引加载成功")
        else:
            logger.warning("[Container] RAG索引未找到，需要上传文档构建")

        logger.info("[Container] 系统启动完成")

    async def shutdown(self) -> None:
        """系统关闭时调用"""
        logger.info("[Container] 系统关闭中...")

        # 关闭RAG引擎
        if self._rag_engine:
            self._rag_engine.close()

        # 关闭向量存储
        if self._vector_store:
            self._vector_store.close()

        logger.info("[Container] 系统关闭完成")


# 全局容器实例
"""
# 这是Lazy Initialization（懒加载）+ 单例模式，所以全局唯一 Container
否则每次请求都会创建新的对象VectorDB、Embedding、LLM Client等，导致内存爆炸、性能崩溃、数据库冲突
"""

_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局容器实例"""
    global _container
    if _container is None:
        _container = Container()
    return _container
