"""
main.py —— FastAPI应用启动入口
================================
职责：创建FastAPI应用、注册路由、配置生命周期
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.container import get_container
from src.api.chat_api import router as chat_router
from src.api.upload_api import router as upload_router
from src.infra.logger import setup_logger

# 配置日志
setup_logger(name="rag_system", level=logging.INFO, log_file="logs/app.log")
logger = logging.getLogger(__name__)


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

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(upload_router)


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
    rag_ready = container.rag_engine().is_ready()

    return {
        "status": "healthy",
        "rag_ready": rag_ready
    }


if __name__ == "__main__":
    import uvicorn

    # unicorn接受浏览器发送的请求，交给fastapi处理
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
