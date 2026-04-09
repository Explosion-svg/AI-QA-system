"""
prompt_builder.py —— Prompt构建器
==================================
职责：构建最终的LLM prompt
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Prompt构建器
    将检索到的上下文和用户问题组合成最终prompt
    """

    def __init__(
        self,
        system_template: Optional[str] = None,
        context_template: Optional[str] = None
    ):
        """
        初始化Prompt构建器

        Args:
            system_template: 系统提示词模板
            context_template: 上下文模板
        """
        self.system_template = system_template or (
            "你是一个智能助手，请根据提供的参考资料回答用户的问题。"
            "如果参考资料不足以回答问题，可以结合自身知识进行补充。"
        )

        self.context_template = context_template or (
            "参考资料:\n{context}\n\n问题: {question}"
        )

        logger.info("[PromptBuilder] 初始化完成")

    def build_prompt(
        self,
        question: str,
        context: str,
        include_system: bool = True
    ) -> str:
        """
        构建完整prompt

        Args:
            question: 用户问题
            context: 检索到的上下文
            include_system: 是否包含系统提示词

        Returns:
            构建好的prompt
        """
        # 构建用户消息
        user_message = self.context_template.format(
            context=context,
            question=question
        )

        # 是否包含系统提示词
        if include_system:
            return f"{self.system_template}\n\n{user_message}"
        else:
            return user_message

    def build_context_string(
        self,
        documents: List[str],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        将文档列表拼接成上下文字符串

        Args:
            documents: 文档内容列表
            separator: 分隔符

        Returns:
            拼接后的上下文
        """
        if not documents:
            return ""

        return separator.join(documents)

    def format_with_sources(
        self,
        question: str,
        context: str,
        sources: List[str]
    ) -> str:
        """
        构建带来源信息的prompt

        Args:
            question: 用户问题
            context: 上下文
            sources: 来源列表

        Returns:
            带来源的prompt
        """
        sources_str = "\n".join([f"- {src}" for src in sources])

        enhanced_context = (
            f"{context}\n\n"
            f"以上信息来源于:\n{sources_str}"
        )

        return self.build_prompt(question, enhanced_context)
