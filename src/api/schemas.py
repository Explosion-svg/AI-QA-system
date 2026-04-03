from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="内容")

class ChatRequest(BaseModel):
    message: str = Field(..., example="什么是 RAG？")
    history: List[ChatMessage] = Field(default_factory=list, description="历史对话记录")
    use_rag: bool = Field(default=True, description="是否启用知识库检索")
    provider: Optional[str] = Field(default=None, description="模型提供商 (openai/deepseek/ollama 等)")
    model: Optional[str] = Field(default=None, description="具体的模型名称")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=2048)

class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI 的回答内容")
    sources: List[str] = Field(default=[], description="参考来源列表")
    status: str = Field(default="success")

# 知识库管理相关模型
class KnowledgeBaseStatus(BaseModel):
    ready: bool = Field(..., description="知识库是否就绪")
    sources: List[str] = Field(default=[], description="已索引的文件列表")
    source_count: int = Field(..., description="文件数量")
    message: str = Field(..., description="状态描述")

class KnowledgeBaseResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果描述")
    chunk_count: Optional[int] = Field(None, description="文档块数量")
    sources: Optional[List[str]] = Field(None, description="文件列表")
