"""
rag_engine.py —— RAG 知识库引擎
==================================
RAG = Retrieval-Augmented Generation（检索增强生成）

原理（3 个阶段）：
  1. 索引阶段（一次性）：
     文档 -> 按段落切片 -> 每片转成「向量」（一组数字，代表语义）-> 存入向量数据库

  2. 检索阶段（每次提问时）：
     用户问题 -> 也转成向量 -> 在数据库里找「最相似」的文档片段

  3. 生成阶段：
     把检索到的片段 + 用户问题一起发给 AI -> AI 基于这些资料回答

生命周期设计：
  RAGEngine 持有一个显式的 chromadb.PersistentClient，
  所有对向量库的读写都通过这个 client 进行。
  调用方（api.py lifespan）负责在服务关闭时调用 engine.close()，
  确保文件句柄在进程退出前被正确释放，彻底避免 Windows [WinError 32] 文件锁问题。

使用的核心库：
- LangChain：文档加载、切片的工具集
- ChromaDB：本地向量数据库（轻量，无需服务器）
- sentence-transformers：本地运行的 Embedding 模型（把文字转成向量）
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, KNOWLEDGE_DIR, RAG_TOP_K

logger = logging.getLogger(__name__)

# 每次写入向量库的 chunk 批量大小
EMBED_BATCH_SIZE = 32

# 向量数据库存放目录（运行后自动生成）
DB_DIR = "vector_db"

# ChromaDB 集合名称（一个 client 可管理多个集合，这里固定用一个）
COLLECTION_NAME = "vector_store"


class RAGEngine:
    """
    RAG 知识库引擎，负责文档的索引构建和语义检索。

    生命周期：
        engine = RAGEngine()
        engine.load_index()          # 启动时加载
        ...                          # 正常使用
        engine.close()               # 关闭时显式释放（在 lifespan 的 shutdown 阶段调用）
    """

    def __init__(self) -> None:
        self.knowledge_dir = Path(KNOWLEDGE_DIR)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        self._db_dir = Path(DB_DIR)

        # Embedding 模型（延迟加载，首次使用时才下载 ~120MB）
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

        # 显式持有的 ChromaDB 持久化客户端（生命周期与 RAGEngine 实例绑定）
        self._chroma_client: Optional[chromadb.PersistentClient] = None

        # LangChain 的 Chroma wrapper（基于 _chroma_client 创建，不自行管理连接）
        self._vectorstore: Optional[Chroma] = None

        # 向量库未加载
        self._loaded: bool = False

    # ==============================================================
    # 生命周期管理
    # ==============================================================

    def _get_client(self) -> chromadb.PersistentClient:
        """
        获取（或创建）ChromaDB 持久化客户端。
        单例：同一个 RAGEngine 实例只持有一个 client，避免多个进程同时写同一目录。
        """
        if self._chroma_client is None:
            self._db_dir.mkdir(parents=True, exist_ok=True)
            """
            self._chroma_client是ChromaDB数据库连接，
            只负责数据库的管理、collection管理和底层存储
            不会做 embedding / 相似度搜索
            """
            self._chroma_client = chromadb.PersistentClient(
                path=str(self._db_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("[RAG] ChromaDB PersistentClient 已创建: %s", self._db_dir)
        return self._chroma_client

    def close(self) -> None:
        """
        程序关闭时，显式释放 ChromaDB 的数据库连接和文件句柄。
        应在服务关闭时（lifespan shutdown 阶段）调用，而不是依赖 GC。
        """
        self._vectorstore = None
        if self._chroma_client is not None:
            try:
                # chromadb 0.4+ PersistentClient 没有 close() 方法，
                # 但将引用置 None 并 GC 即可触发底层 DuckDB/SQLite 连接关闭。
                # 如果未来版本加了 close()，这里会优先调用。
                close_fn = getattr(self._chroma_client, "close", None)
                if callable(close_fn):  # 如果 close_fn 是函数
                    close_fn()
            except Exception as e:
                logger.warning("[RAG] 关闭 ChromaDB client 时出现警告（可忽略）: %s", e)
            finally:
                self._chroma_client = None
        self._loaded = False
        gc.collect()
        logger.info("[RAG] ChromaDB 连接已关闭。")

    # ==============================================================
    # Embedding 模型
    # ==============================================================

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """获取 Embedding 模型（单例，只初始化一次）。"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    # ==============================================================
    # 文本切片
    # ==============================================================

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        """创建文本切片器。优先按段落切，保留语义完整性。"""
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", ".", "!", "?", " ", ""],
        )

    # ==============================================================
    # 文档加载
    # ==============================================================

    def _load_single(self, path: str) -> List[Document]:
        """加载单个文件，返回 Document 列表。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(p))
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(p), encoding="utf-8")
        else:
            raise ValueError(f"不支持的格式: {suffix}")
        return loader.load()

    def load_documents(self, paths: List[str] = None) -> List[Document]:
        """
        批量加载文档。若不传 paths，则加载 knowledge_base/ 目录下的所有文件。
        """
        docs = []
        search_paths = paths or [str(p) for p in self.knowledge_dir.iterdir() if p.is_file()]
        for p in search_paths:
            try:
                docs.extend(self._load_single(p))
                logger.info("[RAG] 已加载: %s", Path(p).name)
            except Exception as e:
                logger.error("[RAG] 加载 %s 失败: %s", Path(p).name, e)
        return docs

    # ==============================================================
    # 索引构建
    # ==============================================================

    def _chunk_generator(self, docs: List[Document]) -> Generator[List[Document], None, None]:
        """将 Document 列表切片后，按 EMBED_BATCH_SIZE 逐批 yield，控制内存占用。"""
        splitter = self._splitter()
        chunks = splitter.split_documents(docs)
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            yield chunks[i: i + EMBED_BATCH_SIZE]

    def _get_vectorstore(self) -> Chroma:
        """
        获取（或创建）LangChain Chroma wrapper。获取 LangChain 的 Chroma 向量库对象
        复用已有的 _chroma_client，不重新建立连接。
        """
        # self._vectorstore是LangChain 的向量库操作封装，就是向量库的操作接口
        if self._vectorstore is None:
            # 复用之前的PersistentClient，不会重复读加载
            client = self._get_client()
            """
            LangChain的 Chroma 封装，提供add_documents(),similarity_search(),delete()
            """
            self._vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self._get_embeddings(),
            )
        return self._vectorstore

    def build_index_batch(self, paths: List[str]) -> Generator[dict, None, None]:
        """
        伪流式 batch 索引构建：Loader → Chunk → Embedding batch → Vector DB。

        每处理完一个 batch 就 yield 进度信息，调用方可实时感知进度。
        yield 格式：
            {"file": str, "batch": int, "batch_chunks": int, "total_chunks": int, "done": bool}
            {"file": str, "error": str, "done": False}          # 单文件失败
            {"done": True, "total_chunks": int}                  # 全部完成

        设计要点：
        - 复用 _chroma_client，不重复建立连接
        - 文件级失败隔离，单个文件出错不影响其他文件
        - 每批 embedding 后立即写库，内存占用恒定
        """
        vectorstore = self._get_vectorstore()
        total_chunks = 0

        for path in paths:
            filename = Path(path).name
            batch_idx = 0
            try:
                docs = self._load_single(path)
                logger.info("[RAG] Loader: 已加载 %s，%d 页/块", filename, len(docs))

                for batch in self._chunk_generator(docs):
                    batch_idx += 1
                    total_chunks += len(batch)

                    vectorstore.add_documents(batch)
                    self._loaded = True

                    logger.info(
                        "[RAG] %s batch#%d: 写入 %d chunks，累计 %d",
                        filename, batch_idx, len(batch), total_chunks,
                    )
                    yield {
                        "file": filename,
                        "batch": batch_idx,
                        "batch_chunks": len(batch),
                        "total_chunks": total_chunks,
                        "done": False,
                    }

            except Exception as e:
                logger.error("[RAG] 处理文件失败 %s: %s", filename, e)
                yield {
                    "file": filename,
                    "batch": batch_idx,
                    "batch_chunks": 0,
                    "total_chunks": total_chunks,
                    "done": False,
                    "error": str(e),
                }

        logger.info("[RAG] 全部索引构建完成，共 %d chunks", total_chunks)
        yield {"file": "", "batch": 0, "batch_chunks": 0, "total_chunks": total_chunks, "done": True}

    def build_index(self, paths: List[str] = None) -> int:
        """
        同步构建向量索引（CLI 使用）。
        返回成功索引的文档块数量。
        """
        docs = self.load_documents(paths)
        if not docs:
            logger.warning("[RAG] 没有可索引的文档。")
            return 0

        splitter = self._splitter()
        chunks = splitter.split_documents(docs)
        logger.info("[RAG] 切片完成，共 %d 块", len(chunks))

        self._get_vectorstore().add_documents(chunks)
        self._loaded = True

        logger.info("[RAG] 索引构建完成，共 %d 块已存入向量数据库。", len(chunks))
        return len(chunks)

    def load_index(self) -> bool:
        """
        连接到已有的向量索引（无需重新 Embedding）。
        若索引目录不存在或集合为空则返回 False。
        """

        if not self._db_dir.exists():
            return False
        try:
            # 获取LangChain 的 Chroma 向量库对象
            vectorstore = self._get_vectorstore()
            # 检查集合内是否有数据
            count = vectorstore._collection.count()

            if count == 0:
                logger.warning("[RAG] 向量库存在但集合为空。")
                # 调试
                # logger.warning("!!!!!! TEST !!!!! 向量库为空")

                return False
            self._loaded = True
            logger.info("[RAG] 已连接到向量索引，共 %d 条记录。", count)
            return True
        except Exception as e:
            logger.error("[RAG] 连接向量索引失败: %s", e)
            return False

    # ==============================================================
    # 检索
    # ==============================================================

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Tuple[Document, float]]:
        """
        语义检索：返回与 query 最相关的 top_k 个文档块及其相似度分数。
        返回值：[(Document, score), ...]，score 越高越相关。
        """
        if not self._loaded or self._vectorstore is None:
            return []
        return self._vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

    def get_context(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """检索后将文档块拼接成上下文字符串，直接供 Prompt 使用。"""
        results = self.retrieve(query, top_k)
        if not results:
            return ""

        """
        metadata里面是这样的
        metadata:{
            "source":"rag.pdf",
            "page":2}
        """
        parts = []
        for doc, score in results:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f" | 第{page + 1}页" if page != "" else ""
            header = f"[来源: {Path(source).name}{page_info}  相似度: {score:.2f}]"
            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n---\n".join(parts)

    def get_context_with_sources(self, query: str, top_k: int = RAG_TOP_K) -> Tuple[str, List[str]]:
        """
        同时返回上下文字符串和来源列表，供前端展示。
        返回值：(rag_context: str, sources: list[str])
        """
        results = self.retrieve(query, top_k)
        if not results:
            return "", []

        parts = []
        sources: List[str] = []
        for doc, score in results:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            name = Path(source).name if source else "未知来源"
            page_info = f" 第{page + 1}页" if page != "" else ""
            entry = f"{name}{page_info}（相似度 {score:.2f}）"
            if entry not in sources:
                sources.append(entry)
            header = f"[来源: {name}{page_info}  相似度: {score:.2f}]"
            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n---\n".join(parts), sources

    # ==============================================================
    # 工具方法
    # ==============================================================

    def is_ready(self) -> bool:
        """索引是否已就绪（可以检索）。"""
        return self._loaded and self._vectorstore is not None

    def list_sources(self) -> List[str]:
        """列出向量库中所有文档的来源文件名（去重）。"""
        if not self.is_ready():
            return []
        try:
            all_meta = self._vectorstore.get()["metadatas"]
            return sorted({Path(m.get("source", "")).name for m in all_meta if m.get("source")})
        except Exception as e:
            logger.warning("[RAG] list_sources 失败: %s", e)
            return []

    def clear_index(self) -> None:
        """
        清空向量数据库。

        正确流程：
          1. 通过已有 client 删除集合（在连接内部操作，不碰文件系统）
          2. 重置 vectorstore wrapper
          3. 关闭并释放 client（让 ChromaDB 自行 flush 并关闭文件句柄）
          4. GC，确保 Python 侧引用归零
          5. 删除磁盘目录（此时句柄已全部释放，Windows 文件锁不再存在）
        """
        # Step 1: 在连接内部删除集合，ChromaDB 自行处理文件写入和关闭
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(COLLECTION_NAME)
                logger.info("[RAG] 集合 '%s' 已删除。", COLLECTION_NAME)
            except Exception as e:
                logger.warning("[RAG] 删除集合失败（可能集合不存在）: %s", e)

        # Step 2-4: 显式关闭连接，释放所有文件句柄
        self.close()

        # Step 5: 此时文件句柄已释放，安全删除目录
        if self._db_dir.exists():
            self._rmtree_with_retry(str(self._db_dir))

        logger.info("[RAG] 向量索引已清空。")

    def _rmtree_with_retry(self, path: str, retries: int = 3, delay: float = 0.5) -> None:
        """
        删除目录，失败时重试。
        专门应对 Windows 上文件句柄释放有短暂延迟的情况。
        """
        for attempt in range(1, retries + 1):
            try:
                shutil.rmtree(path)
                return
            except PermissionError as e:
                if attempt == retries:
                    raise RuntimeError(
                        f"删除 {path} 失败，已重试 {retries} 次。"
                        f"请确认没有其他进程占用该目录后重试。"
                    ) from e
                logger.warning(
                    "[RAG] 目录删除被锁 (第 %d/%d 次)，%.1f s 后重试: %s",
                    attempt, retries, delay, e,
                )
                time.sleep(delay)
                gc.collect()


if __name__ == "__main__":
    """
    测试 RAGEngine
    运行：python -m src.core.rag_engine
    """
    print("=" * 50)
    print("测试 RAGEngine")
    print("=" * 50)

    engine = RAGEngine()

    print("\n[测试 1] 加载已有索引")
    if engine.load_index():
        print("✅ 索引加载成功")
        print(f"索引状态: {engine.is_ready()}")

        print("\n[测试 2] 语义检索")
        results = engine.retrieve("什么是 RAG？", top_k=3)
        print(f"检索到 {len(results)} 个结果")

        print("\n[测试 3] 获取上下文")
        context, sources = engine.get_context_with_sources("RAG 的原理")
        print(f"上下文长度: {len(context)}")
        print(f"来源: {sources}")
    else:
        print("⚠️ 未找到索引，请先通过 Web 或 CLI 构建知识库")

    engine.close()
    print("\n✅ 测试完成")
