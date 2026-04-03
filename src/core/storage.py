"""
storage.py —— 文件存储管理器
================================
职责：文件的持久化存储、流式写入、文件管理
"""

import logging
import hashlib
import aiofiles
from pathlib import Path
from typing import BinaryIO, AsyncIterator
from datetime import datetime

from src.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

# 单个文件最大大小：50MB
MAX_FILE_SIZE = 50 * 1024 * 1024
# 流式写入块大小：1MB
CHUNK_SIZE = 1024 * 1024


class StorageManager:
    """文件存储管理器：负责文件的安全存储和管理"""

    def __init__(self, base_dir: str = KNOWLEDGE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # file_stream 是一个 async generator
    async def save_file_stream(self, filename: str, file_stream: AsyncIterator[bytes]) -> tuple[str, str]:
        """
        流式保存文件到磁盘（防止大文件占满内存）
        :param filename: 文件名
        :param file_stream: 异步文件流
        :return: (保存路径, 文件哈希)
        """
        # 生成唯一文件名（时间戳 + 原文件名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = self._sanitize_filename(filename)
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = self.base_dir / unique_filename

        # 流式写入 + 计算哈希
        file_hash = hashlib.sha256()
        total_size = 0

        try:
            async with aiofiles.open(file_path, "wb") as f:
                async for chunk in file_stream:
                    total_size += len(chunk)

                    # 文件大小限制
                    if total_size > MAX_FILE_SIZE:
                        await f.close()
                        file_path.unlink(missing_ok=True)   # 删除已经写入的文件
                        raise ValueError(f"文件超过最大限制 {MAX_FILE_SIZE / 1024 / 1024}MB")

                    file_hash.update(chunk)
                    await f.write(chunk)    # 按chunk一块块写入磁盘

            logger.info(f"[Storage] 文件已保存: {unique_filename}, 大小: {total_size / 1024:.2f}KB")
            return str(file_path), file_hash.hexdigest()

        except Exception as e:
            # 保存失败时清理文件
            file_path.unlink(missing_ok=True)
            logger.error(f"[Storage] 保存文件失败: {e}")
            raise

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除危险字符"""
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        safe_name = filename
        for char in dangerous_chars:
            safe_name = safe_name.replace(char, '_')
        return safe_name

    def list_files(self) -> list[str]:
        """列出所有已存储的文件"""
        return [str(p) for p in self.base_dir.glob("*") if p.is_file()]

    def delete_file(self, file_path: str) -> bool:
        """删除指定文件"""
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.info(f"[Storage] 文件已删除: {file_path}")
            return True
        except Exception as e:
            logger.error(f"[Storage] 删除文件失败: {e}")
            return False

    def clear_all(self) -> int:
        """清空所有文件"""
        count = 0
        for file_path in self.list_files():
            if self.delete_file(file_path):
                count += 1
        return count
