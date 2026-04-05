"""
upload_service.py —— 文件上传业务服务
=====================================
职责：协调文件上传、解析、索引的完整流程
"""

import logging
from typing import List
from pathlib import Path

from fastapi import UploadFile

from src.rag.rag_engine import RAGEngine
from src.index.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class UploadService:
    """
    文件上传业务服务
    负责处理文档上传的完整业务流程
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        document_loader: DocumentLoader
    ):
        """
        初始化上传服务

        Args:
            rag_engine: RAG引擎
            document_loader: 文档加载器
        """
        self.rag_engine = rag_engine
        self.document_loader = document_loader
        self.storage_dir = Path("knowledge_base")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[UploadService] 初始化完成")

    async def upload_and_index(self, files: List[UploadFile]) -> dict:
        """
        上传文件并构建索引

        Args:
            files: 上传文件列表

        Returns:
            处理结果
        """
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0
        }

        saved_paths = []

        # 1. 保存文件
        for file in files:
            try:
                # 验证格式
                if not self.document_loader.is_supported(file.filename):
                    results["failed"].append({
                        "filename": file.filename,
                        "reason": "不支持的文件格式"
                    })
                    continue

                # 保存文件
                file_path = self.storage_dir / file.filename
                content = await file.read()

                with open(file_path, "wb") as f:
                    f.write(content)

                saved_paths.append(str(file_path))
                results["success"].append({
                    "filename": file.filename,
                    "path": str(file_path)
                })

                logger.info(f"[UploadService] 文件已保存: {file.filename}")

            except Exception as e:
                logger.error(f"[UploadService] 保存文件失败 {file.filename}: {e}")
                results["failed"].append({
                    "filename": file.filename,
                    "reason": str(e)
                })

        # 2. 构建索引
        if saved_paths:
            try:
                total_chunks = self.rag_engine.build_index(saved_paths)
                results["total_chunks"] = total_chunks
                logger.info(f"[UploadService] 索引构建完成: {total_chunks} chunks")

            except Exception as e:
                logger.error(f"[UploadService] 索引构建失败: {e}")
                raise

        return results

    def get_status(self) -> dict:
        """
        获取系统状态

        Returns:
            状态信息
        """
        storage_files = list(self.storage_dir.glob("*"))
        storage_files = [f for f in storage_files if f.is_file()]

        return {
            "storage_files": len(storage_files),
            "rag_ready": self.rag_engine.is_ready(),
            "indexed_sources": self.rag_engine.list_sources() if self.rag_engine.is_ready() else []
        }

    def clear_all(self) -> bool:
        """
        清空所有文件和索引

        Returns:
            是否成功
        """
        try:
            # 清空存储文件
            for file_path in self.storage_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()

            # 清空索引
            self.rag_engine.clear_index()

            logger.info("[UploadService] 已清空所有文件和索引")
            return True

        except Exception as e:
            logger.error(f"[UploadService] 清空失败: {e}")
            return False
