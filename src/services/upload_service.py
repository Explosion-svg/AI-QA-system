"""
upload_service.py —— 文档上传业务服务
=====================================
职责：协调文件上传、解析、索引的完整流程，
      同时作为 RAGEngine 生命周期的唯一管理者。

分层原则：
  API 层只与 UploadService / ChatService 交互，
  不直接感知 RAGEngine（Core 层）的存在。
  RAGEngine 的创建、加载、关闭全部由本 Service 负责。
"""

import logging
from typing import List

from fastapi import UploadFile

from src.core.parser import DocumentParser
from src.core.rag_engine import RAGEngine
from src.core.storage import StorageManager

logger = logging.getLogger(__name__)


class UploadService:
    """
    文档上传服务：处理文档上传的完整业务流程。
    同时作为 RAGEngine 对外的唯一出口，API 层通过本 Service 获取引擎状态，
    不直接持有或操作 RAGEngine 实例。
    """

    def __init__(self, rag_engine: RAGEngine = None):
        self.storage = StorageManager()
        self.parser = DocumentParser()
        self.rag_engine = rag_engine or RAGEngine()

    # ------------------------------------------------------------------
    # 生命周期代理：API 层通过 UploadService 管理引擎，不直接调用 Core 层
    # ------------------------------------------------------------------

    def close(self) -> None:
        """代理 RAGEngine.close()，由 API lifespan shutdown 调用。"""
        if self.rag_engine is not None:
            self.rag_engine.close()
            logger.info("[UploadService] RAGEngine 已关闭。")

    def load_index(self) -> bool:
        """代理 RAGEngine.load_index()，由 API lifespan startup 调用。"""
        return self.rag_engine.load_index() if self.rag_engine else False

    async def upload_and_index(self, files: List[UploadFile]) -> dict:
        """
        上传文件并构建索引（完整流程）
        :param files: FastAPI UploadFile 列表
        :return: 处理结果统计
        """
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0
        }

        saved_paths = []

        # 1. 流式保存文件
        for file in files:
            try:
                # 验证文件格式
                if not self.parser.is_supported(file.filename):
                    results["failed"].append({
                        "filename": file.filename,
                        "reason": "不支持的文件格式"
                    })
                    continue

                # 流式保存（分块读取，防止内存溢出）
                async def file_stream():
                    while chunk := await file.read(1024 * 1024):  # 1MB per chunk
                        yield chunk

                file_path, file_hash = await self.storage.save_file_stream(
                    file.filename,
                    file_stream()
                )

                saved_paths.append(file_path)
                results["success"].append({
                    "filename": file.filename,
                    "path": file_path,
                    "hash": file_hash
                })

            except Exception as e:
                logger.error(f"[UploadService] 保存文件失败 {file.filename}: {e}")
                results["failed"].append({
                    "filename": file.filename,
                    "reason": str(e)
                })

        # 2. 伪流式 batch 构建索引（Loader → Chunk → Embedding batch → Vector DB）
        if saved_paths:
            try:
                for progress in self.rag_engine.build_index_batch(saved_paths):
                    if progress.get("done"):
                        results["total_chunks"] = progress["total_chunks"]
                        logger.info(f"[UploadService] 全部索引构建完成，共 {progress['total_chunks']} chunks")
                    elif "error" in progress:
                        logger.error(
                            f"[UploadService] 文件处理失败 {progress['file']}: {progress['error']}"
                        )
                    else:
                        logger.info(
                            f"[UploadService] {progress['file']} "
                            f"batch#{progress['batch']} 已写入 {progress['batch_chunks']} chunks"
                        )
            except Exception as e:
                logger.error(f"[UploadService] 索引构建失败: {e}")
                raise

        return results

    def get_status(self) -> dict:
        """获取上传系统状态"""
        return {
            "storage_files": len(self.storage.list_files()),
            "rag_ready": self.rag_engine.is_ready(),
            "indexed_sources": self.rag_engine.list_sources() if self.rag_engine.is_ready() else []
        }

    def is_rag_ready(self) -> bool:
        """代理 RAGEngine.is_ready()，供 API 层健康检查使用。"""
        return self.rag_engine.is_ready() if self.rag_engine else False

    def list_sources(self) -> list:
        """代理 RAGEngine.list_sources()，供 API 层构建响应使用。"""
        return self.rag_engine.list_sources() if self.rag_engine else []

    def clear_all(self) -> bool:
        """清空所有文件和索引"""
        try:
            self.storage.clear_all()
            self.rag_engine.clear_index()
            logger.info("[UploadService] 已清空所有文件和索引")
            return True
        except Exception as e:
            logger.error(f"[UploadService] 清空失败: {e}")
            return False
