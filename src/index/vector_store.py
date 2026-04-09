"""
vector_store.py —— 向量存储抽象接口
====================================
职责：定义向量数据库的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from langchain_core.documents import Document


class VectorStore(ABC):
    """
    向量存储抽象基类
    定义向量数据库的标准接口，便于切换不同实现
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量库

        Args:
            documents: 文档列表
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        相似度检索

        Args:
            query: 查询文本
            k: 返回top-k结果
            filter_dict: 过滤条件

        Returns:
            相似文档列表
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        带分数的相似度检索

        Args:
            query: 查询文本
            k: 返回top-k结果

        Returns:
            (文档, 分数)元组列表
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """删除整个集合"""
        pass

    @abstractmethod
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """检查向量库是否为空"""
        pass
