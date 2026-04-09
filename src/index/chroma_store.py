"""
chroma_store.py —— ChromaDB向量存储实现
========================================
职责：ChromaDB的具体实现
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.index.vector_store import VectorStore
from src.infra.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """
    ChromaDB向量存储实现
    """

    def __init__(
        self,
        collection_name: str = "vector_store",
        persist_directory: str = "vector_db",
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        初始化ChromaDB存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_model: embedding模型实例
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or EmbeddingModel()
        self._client: Optional[chromadb.PersistentClient] = None
        self._vectorstore: Optional[Chroma] = None

        logger.info(f"[ChromaStore] 初始化: {persist_directory}/{collection_name}")

    def _get_client(self) -> chromadb.PersistentClient:
        """获取或创建ChromaDB客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"[ChromaStore] ChromaDB客户端已创建")
        return self._client

    def _get_vectorstore(self) -> Chroma:
        """获取或创建向量存储"""
        if self._vectorstore is None:
            client = self._get_client()
            embeddings = self.embedding_model.get_embeddings()

            self._vectorstore = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
            logger.info(f"[ChromaStore] 向量存储已就绪")
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量库"""
        if not documents:
            return

        vectorstore = self._get_vectorstore()
        vectorstore.add_documents(documents)
        logger.info(f"[ChromaStore] 已添加 {len(documents)} 个文档")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """相似度检索"""
        vectorstore = self._get_vectorstore()
        return vectorstore.similarity_search(query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度检索"""
        vectorstore = self._get_vectorstore()
        return vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        """删除集合"""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._vectorstore = None
            logger.info(f"[ChromaStore] 集合已删除: {self.collection_name}")
        except Exception as e:
            logger.error(f"[ChromaStore] 删除集合失败: {e}")
            raise

    def get_collection_count(self) -> int:
        """获取文档数量"""
        try:
            client = self._get_client()
            collection = client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def is_empty(self) -> bool:
        """检查是否为空"""
        return self.get_collection_count() == 0

    def close(self) -> None:
        """关闭连接"""
        self._vectorstore = None
        if self._client is not None:
            try:
                close_fn = getattr(self._client, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                logger.warning(f"[ChromaStore] 关闭客户端警告: {e}")
            finally:
                self._client = None
        logger.info("[ChromaStore] 连接已关闭")
