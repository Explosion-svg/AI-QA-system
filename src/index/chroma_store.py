"""
chroma_store.py —— ChromaDB 向量存储实现
=========================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.index.vector_store import VectorStore
from src.infra.config import VECTOR_UPSERT_BATCH_SIZE
from src.infra.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """基于 Chroma 原生 collection 的向量存储实现。"""

    def __init__(
        self,
        collection_name: str = "knowledge_chunks",
        persist_directory: str = "vector_db",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or EmbeddingModel()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

        logger.info(
            "[ChromaStore] 初始化 collection=%s persist_directory=%s",
            collection_name,
            persist_directory,
        )

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def add_documents(self, documents: List[Document]) -> None:
        self.upsert_documents(documents)

    def upsert_documents(self, documents: List[Document]) -> int:
        """
        把文档分批次 → 生成向量 → 存入 Chroma 向量库（存在就更新，不存在就插入）
        :param documents:
        :return:
        """
        if not documents:
            return 0

        collection = self._get_collection()
        # 分批处理
        for start in range(0, len(documents), VECTOR_UPSERT_BATCH_SIZE):
            batch = documents[start:start + VECTOR_UPSERT_BATCH_SIZE]
            ids = []
            texts = []
            metadatas = []

            # 拆分每个chunk数据
            for document in batch:
                chunk_id = document.metadata.get("chunk_id")
                if not chunk_id:
                    raise ValueError("Document metadata 缺少 chunk_id，无法 upsert")

                ids.append(chunk_id)
                texts.append(document.page_content)
                metadatas.append(document.metadata)

            # 批量生成向量
            embeddings = self.embedding_model.embed_documents(texts)
            # 写入向量库（有则更新，无则插入）
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        logger.info("[ChromaStore] 已 upsert %d 个文档块", len(documents))
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None,
    ) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k, filter_dict=filter_dict)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        if k <= 0 or self.is_empty():
            return []

        collection = self._get_collection()
        query_embedding = self.embedding_model.embed_text(query)
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[Tuple[Document, float]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            results.append((Document(page_content=text, metadata=metadata or {}), float(distance)))
        return results

    def get_all_documents(self) -> List[Document]:
        if self.is_empty():
            return []

        collection = self._get_collection()
        raw = collection.get(include=["documents", "metadatas"])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(documents, metadatas)
        ]

    def list_sources(self) -> List[str]:
        if self.is_empty():
            return []

        collection = self._get_collection()
        raw = collection.get(include=["metadatas"])
        sources = {
            metadata.get("source")
            for metadata in raw.get("metadatas", [])
            if metadata and metadata.get("source")
        }
        return sorted(sources)

    def delete_by_source(self, sources: Sequence[str]) -> int:
        collection = self._get_collection()
        deleted = 0
        for source in {item for item in sources if item}:
            existing = collection.get(where={"source": source})
            ids = existing.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                deleted += len(ids)
        if deleted:
            logger.info("[ChromaStore] 已按来源删除 %d 个文档块", deleted)
        return deleted

    def delete_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            logger.debug("[ChromaStore] 集合 %s 不存在，跳过删除", self.collection_name)
        finally:
            self._collection = None

    def get_collection_count(self) -> int:
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0

    def is_empty(self) -> bool:
        return self.get_collection_count() == 0

    def close(self) -> None:
        self._collection = None
        if self._client is not None:
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.warning("[ChromaStore] 关闭客户端失败: %s", exc)
            self._client = None
