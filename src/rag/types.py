"""
types.py —— RAG 检索过程中的统一数据结构
===========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document


@dataclass(slots=True)
class RetrievedChunk:
    """统一的检索候选结构，贯穿召回、融合、重排和过滤流程。"""

    document: Document
    chunk_id: str
    source: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def page_content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata
