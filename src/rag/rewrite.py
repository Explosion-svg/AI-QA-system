"""
一种检索策略
query_rewriter.py —— 查询改写模块
====================================
职责：将用户原始问题改写成多个检索友好的查询，提升召回率。

策略：
  1. 原始查询保留
  2. 关键词提取（去停用词）
  3. 同义词扩展（可选，需要词典）
  4. 多角度改写（可选，需要 LLM）

工程级设计：
  - 可插拔：每个策略独立，可按需启用
  - 可配置：通过参数控制改写强度
  - 轻量化：默认只做关键词提取，不依赖外部 LLM
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# 中文停用词（简化版，生产环境建议从文件加载）
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "什么", "怎么", "为什么", "哪里", "如何", "吗", "呢",
}


class QueryRewriter:
    """
    查询改写器：生成多个检索友好的查询变体。

    使用示例：
        rewriter = QueryRewriter()
        queries = rewriter.rewrite("如何使用 RAG 提升问答准确率？")
        # 返回: ["如何使用 RAG 提升问答准确率？", "RAG 提升 问答 准确率", ...]
    """

    def __init__(self, enable_keyword_extract: bool = True, enable_synonym: bool = False):
        """
        :param enable_keyword_extract: 是否启用关键词提取
        :param enable_synonym: 是否启用同义词扩展（需要词典，暂未实现）
        """
        self.enable_keyword_extract = enable_keyword_extract
        self.enable_synonym = enable_synonym

    def rewrite(self, query: str) -> List[str]:
        """
        改写查询，返回多个变体。

        :param query: 原始查询
        :return: 查询列表，第一个永远是原始查询
        """
        queries = [query]  # 原始查询保留

        if self.enable_keyword_extract:
            keywords = self._extract_keywords(query)
            if keywords and keywords != query:
                queries.append(keywords)

        logger.debug("[QueryRewriter] 原始查询: %s", query)
        logger.debug("[QueryRewriter] 改写后: %s", queries)
        return queries

    def _extract_keywords(self, text: str) -> str:
        """
        提取关键词：去除停用词、标点，保留核心词汇。

        简化实现：分词 + 停用词过滤。
        生产环境建议用 jieba 分词 + 自定义词典。
        """
        # 去除标点符号
        text = re.sub(r"[^\w\s]", " ", text)
        # 简单按空格分词（中文会按字分，英文按词分）
        words = text.split()
        # 过滤停用词和单字符
        keywords = [w for w in words if w not in STOPWORDS and len(w) > 1]
        return " ".join(keywords) if keywords else text


if __name__ == "__main__":
    # 测试
    rewriter = QueryRewriter()
    result = rewriter.rewrite("如何使用 RAG 提升问答的准确率？")
    print("改写结果:")
    for i, q in enumerate(result, 1):
        print(f"  {i}. {q}")
