"""
rewrite.py —— 查询改写模块
===========================
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "什么", "怎么", "为什么", "哪里", "如何", "吗", "呢",
    "请问", "一下", "关于",
}
# 切分中文词和英文词
CHINESE_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_./:-]*")


class QueryRewriter:
    """生成更适合检索的 query 变体。"""

    def __init__(self, enable_keyword_extract: bool = True):
        self.enable_keyword_extract = enable_keyword_extract

    def rewrite(self, query: str) -> List[str]:
        normalized = self._normalize_query(query)
        queries: List[str] = [normalized]

        if self.enable_keyword_extract:
            keyword_query = self._extract_keywords(normalized)
            if keyword_query and keyword_query != normalized:
                queries.append(keyword_query)

        unique_queries = []
        seen = set()
        for item in queries:
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(key)

        logger.debug("[QueryRewriter] 原始查询=%s 改写结果=%s", query, unique_queries)
        return unique_queries

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        lowered = text.lower()

        for token in LATIN_TOKEN_RE.findall(lowered):
            if token not in STOPWORDS:
                tokens.append(token)

        for block in CHINESE_TOKEN_RE.findall(lowered):
            if block in STOPWORDS:
                continue

            chars = [char for char in block if char.strip()]
            # 单字
            tokens.extend(chars)
            # 双字
            tokens.extend(
                block[index:index + 2]
                for index in range(len(block) - 1)
            )
            # 全部
            if len(block) >= 3:
                tokens.append(block)

        return [token for token in tokens if token and token not in STOPWORDS]

    def _extract_keywords(self, query: str) -> str:
        keywords: List[str] = []
        lowered = query.lower()

        for token in LATIN_TOKEN_RE.findall(lowered):
            if token not in STOPWORDS:
                keywords.append(token)

        for block in CHINESE_TOKEN_RE.findall(lowered):
            cleaned = block
            for stopword in sorted(STOPWORDS, key=len, reverse=True):
                cleaned = cleaned.replace(stopword, " ")
            parts = [part.strip() for part in cleaned.split() if len(part.strip()) >= 2]
            if parts:
                keywords.extend(parts)
            elif len(block) >= 2:
                keywords.append(block)

        # 去重
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return " ".join(unique_keywords[:8])

    @staticmethod
    def _normalize_query(query: str) -> str:
        query = re.sub(r"\s+", " ", query).strip()
        return query
