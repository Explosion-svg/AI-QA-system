"""
parser.py —— 文档解析器
========================
职责：解析不同格式的文档，提取文本内容
"""

import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}


class DocumentParser:
    """文档解析器：支持多种格式的文档解析"""

    @staticmethod
    def is_supported(filename: str) -> bool:
        """检查文件格式是否支持"""
        return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def parse(file_path: str) -> List[Document]:
        """
        解析文档，提取文本内容
        :param file_path: 文件路径
        :return: Document 列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix in (".docx", ".doc"):
                loader = Docx2txtLoader(str(path))
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")

            docs = loader.load()
            logger.info(f"[Parser] 解析成功: {path.name}, 页数/块数: {len(docs)}")
            return docs

        except Exception as e:
            logger.error(f"[Parser] 解析失败 {path.name}: {e}")
            raise
