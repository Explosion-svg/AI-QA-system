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

好比：你问问题时，先去图书馆找相关页面，然后带着这些页面去问专家。

使用的核心库：
- LangChain：文档加载、切片的工具集
- ChromaDB：本地向量数据库（轻量，无需服务器）
- sentence-transformers：本地运行的 Embedding 模型（把文字转成向量）
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import KNOWLEDGE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K

# 向量数据库存放目录（运行后自动生成）
DB_DIR = "vector_db"


class RAGEngine:
    """RAG 知识库引擎，负责文档的索引构建和语义检索。"""

    def __init__(self):
        self.knowledge_dir = Path(KNOWLEDGE_DIR)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)  # 自动创建知识库目录
        self.embeddings = None      # Embedding 模型（延迟加载，首次使用时才下载）
        self.vectorstore = None     # ChromaDB 向量数据库实例
        self._loaded = False        # 标记索引是否已加载

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        获取 Embedding 模型（单例模式，只初始化一次）。
        首次调用会从 HuggingFace 下载模型文件（~120MB），之后从本地缓存加载。
        """
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},              # 用 CPU 运行（轻量化）
                encode_kwargs={"normalize_embeddings": True}, # 归一化向量，提升检索精度
            )
        return self.embeddings

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        """
        创建文本切片器。
        RecursiveCharacterTextSplitter 会优先按段落（\n\n）切，
        切不够小再按句子（。！？），最后按字符切。
        这样能保留更多语义完整性。
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,       # 每块最大字符数
            chunk_overlap=CHUNK_OVERLAP, # 相邻块重叠字符数（保证边界处语义连贯）
            separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", ".", "!", "?", " ", ""],
            # 分隔符优先级：段落 > 换行 > 中文句号/感叹号/问号 > 英文标点 > 空格 > 字符
        )

    # ==============================================================
    # 文档加载
    # ==============================================================

    def load_documents(self, paths: List[str] = None) -> List[Document]:
        """
        从文件路径加载文档，自动识别格式。
        若不传 paths，则加载 knowledge_base/ 目录下的所有文件。

        支持格式：
        - .txt / .md  -> TextLoader（直接读取文本）
        - .pdf        -> PyPDFLoader（逐页解析）
        - .docx       -> Docx2txtLoader（提取正文文字）
        """
        docs = []
        search_paths = paths or [str(p) for p in self.knowledge_dir.iterdir() if p.is_file()]
        for p in search_paths:
            p = Path(p)
            if not p.exists():
                continue
            try:
                if p.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(p))
                elif p.suffix.lower() in (".docx", ".doc"):
                    loader = Docx2txtLoader(str(p))
                elif p.suffix.lower() in (".txt", ".md"):
                    loader = TextLoader(str(p), encoding="utf-8")
                else:
                    continue   # 不支持的格式直接跳过
                docs.extend(loader.load())
                print(f"[RAG] 已加载: {p.name}")
            except Exception as e:
                print(f"[RAG] 加载 {p.name} 失败: {e}")
        return docs

    def build_from_uploads(self, uploaded_files) -> int:
        """
        Streamlit 专用：接收 st.file_uploader 返回的上传文件列表，
        保存到 knowledge_base/ 目录后构建索引。
        """
        saved = []
        for f in uploaded_files:
            dest = self.knowledge_dir / f.name
            with open(dest, "wb") as out:
                out.write(f.read())   # 把上传的字节流写入文件
            saved.append(str(dest))
            print(f"[RAG] 已保存上传文件: {f.name}")
        return self.build_index(saved)

    # ==============================================================
    # 索引构建
    # ==============================================================

    def build_index(self, paths: List[str] = None) -> int:
        """
        构建（或增量更新）向量索引。
        步骤：加载文档 -> 切片 -> Embedding -> 存入 ChromaDB
        返回成功索引的文档块数量。
        """
        docs = self.load_documents(paths)
        if not docs:
            print("[RAG] 没有可索引的文档。")
            return 0

        splitter = self._splitter()
        chunks = splitter.split_documents(docs)
        print(f"[RAG] 切片完成，共 {len(chunks)} 块")

        embeddings = self._get_embeddings()

        # 若数据库已存在则追加，否则新建
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_DIR,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.persist()  # 持久化到磁盘
        self._loaded = True
        print(f"[RAG] 索引构建完成，共 {len(chunks)} 块已存入向量数据库。")
        return len(chunks)

    def load_index(self) -> bool:
        """
        从磁盘加载已有的向量索引（无需重新 Embedding）。
        若索引不存在则返回 False。
        """
        if not Path(DB_DIR).exists():
            return False
        try:
            self.vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self._get_embeddings(),
            )
            self._loaded = True
            print("[RAG] 已从磁盘加载向量索引。")
            return True
        except Exception as e:
            print(f"[RAG] 加载索引失败: {e}")
            return False

    # ==============================================================
    # 检索
    # ==============================================================

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Tuple[Document, float]]:
        """
        语义检索：返回与 query 最相关的 top_k 个文档块及其相似度分数。
        返回值：[(Document, score), ...]，score 越高越相关（余弦相似度）。
        """
        if not self._loaded or self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return results

    def get_context(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """
        检索后将文档块拼接成上下文字符串，直接供 Prompt 使用。
        格式：
          [来源: xxx.pdf | 第2页]
          片段内容...
          ---
          [来源: ...]
          ...
        """
        results = self.retrieve(query, top_k)
        if not results:
            return ""

        parts = []
        for doc, score in results:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f" | 第{page + 1}页" if page != "" else ""
            header = f"[来源: {Path(source).name}{page_info}  相似度: {score:.2f}]"
            parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n---\n".join(parts)

    def get_context_with_sources(self, query: str, top_k: int = RAG_TOP_K):
        """
        同时返回上下文字符串和来源列表，供前端展示。
        避免 app.py 调用两次 retrieve()。
        返回值：(rag_context: str, sources: list)
        """
        results = self.retrieve(query, top_k)
        if not results:
            return "", []

        parts = []
        sources = []
        for doc, score in results:
            source = doc.metadata.get("source", "")
            page   = doc.metadata.get("page", "")
            name   = Path(source).name if source else "未知来源"
            page_info = f" 第{page + 1}页" if page != "" else ""
            entry  = f"{name}{page_info}（相似度 {score:.2f}）"
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
        return self._loaded and self.vectorstore is not None

    def list_sources(self) -> List[str]:
        """列出向量库中所有文档的来源文件名（去重）。"""
        if not self.is_ready():
            return []
        try:
            all_meta = self.vectorstore.get()["metadatas"]
            sources = sorted({Path(m.get("source", "")).name for m in all_meta if m.get("source")})
            return sources
        except Exception:
            return []

    def clear_index(self) -> None:
        """清空向量数据库（慎用）。"""
        import shutil
        if Path(DB_DIR).exists():
            shutil.rmtree(DB_DIR)
        self.vectorstore = None
        self._loaded = False
        print("[RAG] 向量索引已清空。")
