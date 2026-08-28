"""
main.py —— FastAPI应用启动入口
================================
职责：创建FastAPI应用、注册路由、配置生命周期
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from src.container import get_container
from src.api.chat_api import router as chat_router
from src.api.upload_api import router as upload_router
from src.infra.config import API_AUTH_TOKEN, CORS_ORIGINS
from src.infra.logger import setup_logger

# 配置日志
setup_logger(name="rag_system", level=logging.INFO, log_file="logs/app.log")
logger = logging.getLogger(__name__)

# ============================================================
# API 鉴权（配置了 API_AUTH_TOKEN 时，要求请求头 X-API-Token 匹配）
# ============================================================
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)


async def verify_token(token: Optional[str] = Security(api_key_header)) -> None:
    """配置了 API_AUTH_TOKEN 时，要求请求头 X-API-Token 与之匹配。"""
    if API_AUTH_TOKEN and token != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="未授权")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    Args:
        app: FastAPI应用实例
    """
    # 启动
    logger.info("=" * 60)
    logger.info("RAG系统启动中...")
    logger.info("=" * 60)

    container = get_container()
    await container.startup()

    logger.info("=" * 60)
    logger.info("RAG系统启动完成")
    logger.info("=" * 60)

    yield

    # 关闭
    logger.info("=" * 60)
    logger.info("RAG系统关闭中...")
    logger.info("=" * 60)

    await container.shutdown()

    logger.info("=" * 60)
    logger.info("RAG系统已关闭")
    logger.info("=" * 60)


# 创建FastAPI应用
app = FastAPI(
    title="RAG Knowledge Base API",
    description="基于RAG的智能问答系统API",
    version="2.0.0",
    lifespan=lifespan
)

# 配置CORS（显式白名单，不用通配符；不使用 Cookie 凭证，故省略 allow_credentials）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（除根路径 / 与 /health 外，全部走 Token 鉴权）
app.include_router(chat_router, dependencies=[Depends(verify_token)])
app.include_router(upload_router, dependencies=[Depends(verify_token)])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG Knowledge Base API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    container = get_container()
    rag_engine = container.rag_engine()
    rag_ready = rag_engine.is_ready()
    # list_sources 会全量拉取 metadata，放入线程池避免卡住事件循环
    sources = await asyncio.to_thread(rag_engine.list_sources) if rag_ready else []

    return {
        "status": "healthy",
        "rag_ready": rag_ready,
        "source_count": len(sources),
    }


if __name__ == "__main__":
    import uvicorn

    # uvicorn 接受浏览器发送的请求，交给 fastapi 处理
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "").lower() in ("1", "true"),
        log_level="info"
    )
