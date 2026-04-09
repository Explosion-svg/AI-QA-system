"""
context_filter.py —— 上下文过滤与组装前筛选
============================================
"""

from __future__ import annotations

import logging
from typing import List, Set

from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class ContextFilter:
    """对重排后的结果做去重、截断和质量过滤。"""

    def __init__(
        self,
        min_length: int = 80,
        max_total_length: int = 4200,
        max_per_source: int = 3,
    ):
        self.min_length = min_length
        self.max_total_length = max_total_length
        self.max_per_source = max_per_source

    def filter(self, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []

        filtered: List[RetrievedChunk] = []
        total_length = 0
        seen_hashes: Set[str] = set()
        per_source_count: dict[str, int] = {}

        for chunk in chunks:
            content = chunk.page_content.strip()
            if len(content) < self.min_length:
                continue

            content_hash = chunk.metadata.get("content_hash")
            if content_hash and content_hash in seen_hashes:
                continue

            source = chunk.source
            if per_source_count.get(source, 0) >= self.max_per_source:
                continue

            if total_length + len(content) > self.max_total_length:
                break

            filtered.append(chunk)
            per_source_count[source] = per_source_count.get(source, 0) + 1
            total_length += len(content)
            if content_hash:
                seen_hashes.add(content_hash)

            if len(filtered) >= top_k:
                break

        logger.info(
            "[ContextFilter] 过滤完成 input=%d output=%d total_chars=%d",
            len(chunks),
            len(filtered),
            total_length,
        )
        return filtered
