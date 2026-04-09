"""
chat_api.py —— 聊天API接口
===========================
职责：处理聊天HTTP请求，参数校验，调用ChatService
"""

import logging
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.container import get_container
from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ============================================================
# 请求/响应模型，schemas
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    use_rag: bool = Field(True, description="是否使用RAG")
    provider: Optional[str] = Field(None, description="LLM提供商")
    model: Optional[str] = Field(None, description="模型名称")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(2048, ge=1, le=8192, description="最大token数")


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(..., description="AI回答")
    sources: List[str] = Field(default_factory=list, description="参考来源")
    session_id: Optional[str] = Field(None, description="会话ID")


class KnowledgeBaseStatus(BaseModel):
    """知识库状态"""
    ready: bool = Field(..., description="是否就绪")
    source_count: int = Field(..., description="文档数量")
    sources: List[str] = Field(default_factory=list, description="文档列表")
    message: str = Field(..., description="状态消息")


# ============================================================
# 依赖注入,解耦 + 测试性
# 核心思想:API 不负责创建对象
# ============================================================

def get_chat_service() -> ChatService:
    """获取ChatService实例"""
    return get_container().chat_service()


# ============================================================
# API端点
# ============================================================

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    聊天接口

    Args:
        request: 聊天请求
        chat_service: 聊天服务

    Returns:
        聊天响应
    """
    try:
        logger.info(f"[ChatAPI] 收到聊天请求: {request.message[:50]}...")

        # 调用业务层
        answer, sources = await chat_service.chat(
            message=request.message,
            session_id=request.session_id,
            use_rag=request.use_rag,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"[ChatAPI] 聊天失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@router.get("/status", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status(
    chat_service: ChatService = Depends(get_chat_service)
) -> KnowledgeBaseStatus:
    """
    获取知识库状态

    Args:
        chat_service: 聊天服务

    Returns:
        知识库状态
    """
    try:
        status = chat_service.get_knowledge_base_status()
        return KnowledgeBaseStatus(**status)

    except Exception as e:
        logger.error(f"[ChatAPI] 获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict:
    """
    获取会话历史

    Args:
        session_id: 会话ID
        chat_service: 聊天服务

    Returns:
        历史记录
    """
    try:
        history = chat_service.get_history(session_id)
        return {"session_id": session_id, "history": history}

    except Exception as e:
        logger.error(f"[ChatAPI] 获取历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")


@router.delete("/history/{session_id}")
async def clear_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict:
    """
    清空会话历史

    Args:
        session_id: 会话ID
        chat_service: 聊天服务

    Returns:
        操作结果
    """
    try:
        chat_service.clear_history(session_id)
        return {"message": "历史已清空", "session_id": session_id}

    except Exception as e:
        logger.error(f"[ChatAPI] 清空历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空历史失败: {str(e)}")
