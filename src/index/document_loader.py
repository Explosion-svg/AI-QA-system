"""
document_loader.py —— 文档加载器
==================================
职责：加载不同格式的文档并切分成chunks
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    文档加载器
    支持PDF、TXT、DOCX格式
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化文档加载器

        Args:
            chunk_size: 每个chunk的大小
            chunk_overlap: chunk之间的重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )

        logger.info(
            f"[DocumentLoader] 初始化: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def is_supported(self, filename: str) -> bool:
        """
        检查文件格式是否支持

        Args:
            filename: 文件名

        Returns:
            是否支持
        """
        return Path(filename).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文档列表

        Raises:
            ValueError: 不支持的文件格式
            Exception: 加载失败
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not self.is_supported(path.name):
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        logger.info(f"[DocumentLoader] 加载文件: {path.name}")

        try:
            # 根据文件类型选择加载器
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            elif suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(path))
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")

            # 加载文档
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata["source"] = path.name
                doc.metadata["file_path"] = str(path)

            logger.info(f"[DocumentLoader] 加载成功: {len(documents)} 个原始文档")
            return documents

        except Exception as e:
            logger.error(f"[DocumentLoader] 加载失败 {path.name}: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档为chunks

        Args:
            documents: 原始文档列表

        Returns:
            切分后的chunk列表
        """
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            f"[DocumentLoader] 文档切分完成: "
            f"{len(documents)} 个文档 -> {len(chunks)} 个chunks"
        )
        return chunks

    def load_and_split(self, file_path: str) -> List[Document]:
        """
        加载文件并切分

        Args:
            file_path: 文件路径

        Returns:
            切分后的chunk列表
        """
        documents = self.load_file(file_path)
        return self.split_documents(documents)

    def load_directory(
        self,
        directory: str,
        recursive: bool = False
    ) -> List[Document]:
        """
        加载目录下的所有支持文件

        Args:
            directory: 目录路径
            recursive: 是否递归子目录

        Returns:
            所有文档的chunk列表
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"无效的目录: {directory}")

        all_chunks = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and self.is_supported(file_path.name):
                try:
                    chunks = self.load_and_split(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"[DocumentLoader] 跳过文件 {file_path.name}: {e}")

        logger.info(
            f"[DocumentLoader] 目录加载完成: "
            f"{directory}, 共 {len(all_chunks)} 个chunks"
        )
        return all_chunks
