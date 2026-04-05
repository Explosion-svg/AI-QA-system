"""
context_filter.py —— 上下文过滤器
==================================
职责：对重排序后的文档进行质量过滤、去重、截断。

过滤策略：
  1. 去重：相似内容只保留一份
  2. 质量过滤：过滤过短、无意义的文档
  3. 长度截断：控制总 token 数，避免超出 LLM 上下文窗口

工程级设计：
  - 可配置阈值
  - 支持多种去重策略（精确匹配、模糊匹配）
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextFilter:
    """
    上下文过滤器。

    使用示例：
        filter = ContextFilter(min_length=50, max_total_length=2000)
        filtered = filter.filter(reranked_docs)
    """

    def __init__(
        self,
        min_length: int = 50,
        max_total_length: int = 4000,
        similarity_threshold: float = 0.9,
    ):
        """
        :param min_length: 最小文档长度（字符数）
        :param max_total_length: 最大总长度（字符数）
        :param similarity_threshold: 去重相似度阈值（0-1）
        """
        self.min_length = min_length
        self.max_total_length = max_total_length
        self.similarity_threshold = similarity_threshold

    def filter(
        self, documents: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        过滤文档。

        :param documents: 输入文档 [(doc, score), ...]
        :return: 过滤后的文档
        """
        if not documents:
            return []

        filtered = []
        total_length = 0
        seen_contents = []

        for doc, score in documents:
            content = doc.page_content.strip()

            # 1. 质量过滤：过短
            if len(content) < self.min_length:
                continue

            # 2. 去重：与已有文档相似度过高
            if self._is_duplicate(content, seen_contents):
                continue

            # 3. 长度截断：累计长度超限
            if total_length + len(content) > self.max_total_length:
                break

            filtered.append((doc, score))
            seen_contents.append(content)
            total_length += len(content)

        logger.debug(
            "[ContextFilter] 输入 %d，输出 %d，总长度 %d",
            len(documents),
            len(filtered),
            total_length,
        )
        return filtered

    def _is_duplicate(self, content: str, seen: List[str]) -> bool:
        """简单去重：检查是否与已有内容高度重复。"""
        for s in seen:
            if self._similarity(content, s) > self.similarity_threshold:
                return True
        return False

    def _similarity(self, a: str, b: str) -> float:
        """计算两个字符串的相似度（简化版：Jaccard）。"""
        set_a = set(a[:200])  # 只比较前 200 字符
        set_b = set(b[:200])
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
