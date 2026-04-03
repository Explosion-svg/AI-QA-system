"""
api.py —— FastAPI 后端入口
===========================
分层原则：
  API 层只与 Service 层交互（ChatService / UploadService），
  不直接感知任何 Core 层（RAGEngine / LLMClient / StorageManager）的存在。

  RAGEngine 的创建、加载、关闭全部委托给 UploadService 管理，
  API 层通过 UploadService 暴露的代理方法操作。

运行方式：
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ChatRequest, ChatResponse, KnowledgeBaseResponse, KnowledgeBaseStatus
from src.config import setup_logging
from src.services.chat_service import ChatService
from src.services.upload_service import UploadService

setup_logging()
logger = logging.getLogger(__name__)

# 全局服务实例（只在 lifespan 中写入，其余地方只读）
state: dict = {}


# ==============================================================
# 生命周期管理
# ==============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    startup：通过 Service 层初始化所有资源。
    shutdown：通过 Service 层释放所有资源（包括 RAGEngine 文件句柄）。
    API 层不直接接触任何 Core 层对象。
    """
    logger.info("🚀 正在启动 API 服务...")

    try:
        # UploadService 负责创建和持有 RAGEngine
        upload_service = UploadService()
        if upload_service.load_index():
            logger.info("✅ 知识库索引加载成功")
        else:
            logger.warning("⚠️ 未找到已有知识库索引，请先通过 API 上传文件构建")

        # ChatService 共享同一个 RAGEngine 实例（通过 UploadService 获取）
        chat_service = ChatService(rag_engine=upload_service.rag_engine)

        state["upload_service"] = upload_service
        state["chat_service"] = chat_service

    except Exception as e:
        logger.error("❌ 初始化失败: %s", e)
        # 即使失败也保证 API 可以启动，各端点会返回 503
        state["upload_service"] = UploadService()
        state["chat_service"] = ChatService()

    yield  # API 开始运行

    # shutdown：通过 Service 层释放资源，API 层不直接调用 Core 层
    logger.info("🛑 正在关闭 API 服务...")
    upload_service: UploadService = state.get("upload_service")
    if upload_service is not None:
        upload_service.close()
    state.clear()


# ==============================================================
# FastAPI 应用
# ==============================================================

app = FastAPI(
    title="AI QA System API",
    description="基于 RAG 的 AI 问答系统后端接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================
# 对话接口
# ==============================================================

@app.post("/chat", response_model=ChatResponse, summary="发起 AI 对话")
async def chat_endpoint(req: ChatRequest):
    """核心对话接口：支持 RAG 检索和多轮历史对话。"""
    try:
        chat_service: ChatService = state.get("chat_service")
        if not chat_service:
            raise HTTPException(status_code=503, detail="服务未就绪")

        history_list = [m.model_dump() for m in req.history]
        answer, sources = chat_service.chat(
            message=req.message,
            history=history_list,
            use_rag=req.use_rag,
            provider=req.provider,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatResponse(answer=answer, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("处理对话请求时出错: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# ==============================================================
# 健康检查
# ==============================================================

@app.get("/health", summary="健康检查")
async def health_check():
    """返回服务状态和知识库就绪状态。由 Service 层提供，API 层不访问 Core 层。"""
    upload_service: UploadService = state.get("upload_service")
    rag_ready = upload_service.is_rag_ready() if upload_service else False
    return {"status": "ok", "rag_ready": rag_ready}


# ==============================================================
# 知识库管理接口
# ==============================================================

@app.post("/knowledge-base/upload", response_model=KnowledgeBaseResponse, summary="上传文件并构建知识库")
async def upload_and_build(files: List[UploadFile] = File(...)):
    """上传文件并构建知识库索引（追加模式）。支持 txt、pdf、md、docx 格式。"""
    try:
        upload_service: UploadService = state.get("upload_service")
        if not upload_service:
            raise HTTPException(status_code=503, detail="服务未就绪")

        results = await upload_service.upload_and_index(files)

        success_count = len(results["success"])
        failed_count = len(results["failed"])
        if failed_count > 0:
            logger.warning("[API] 部分文件上传失败: %s", results["failed"])

        return KnowledgeBaseResponse(
            success=success_count > 0,
            message=f"成功: {success_count}, 失败: {failed_count}, 文档块: {results['total_chunks']}",
            chunk_count=results["total_chunks"],
            sources=upload_service.list_sources(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] 上传文件失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-base/status", response_model=KnowledgeBaseStatus, summary="获取知识库状态")
async def get_kb_status():
    """获取知识库当前状态。"""
    upload_service: UploadService = state.get("upload_service")
    if not upload_service:
        return KnowledgeBaseStatus(ready=False, sources=[], source_count=0, message="服务未初始化")

    status = upload_service.get_status()
    return KnowledgeBaseStatus(
        ready=status["rag_ready"],
        sources=status["indexed_sources"],
        source_count=len(status["indexed_sources"]),
        message="知识库已就绪" if status["rag_ready"] else "知识库未构建",
    )


@app.delete("/knowledge-base/clear", response_model=KnowledgeBaseResponse, summary="清空知识库")
async def clear_kb():
    """清空知识库索引和所有文件。"""
    try:
        upload_service: UploadService = state.get("upload_service")
        if not upload_service:
            raise HTTPException(status_code=503, detail="服务未就绪")

        success = upload_service.clear_all()
        return KnowledgeBaseResponse(
            success=success,
            message="知识库已清空" if success else "清空失败",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] 清空知识库失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge-base/rebuild", response_model=KnowledgeBaseResponse, summary="重建知识库")
async def rebuild_kb(files: List[UploadFile] = File(...)):
    """清空现有索引，重新上传文件并构建。"""
    try:
        upload_service: UploadService = state.get("upload_service")
        if not upload_service:
            raise HTTPException(status_code=503, detail="服务未就绪")

        upload_service.clear_all()
        results = await upload_service.upload_and_index(files)

        success_count = len(results["success"])
        failed_count = len(results["failed"])

        return KnowledgeBaseResponse(
            success=success_count > 0,
            message=f"知识库已重建，成功: {success_count}, 失败: {failed_count}",
            chunk_count=results["total_chunks"],
            sources=upload_service.list_sources(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] 重建知识库失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
