"""
upload_api.py —— 文件上传API接口
=================================
职责：处理文件上传HTTP请求，调用UploadService
"""

import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field

from src.container import get_container
from src.services.upload_service import UploadService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


# ============================================================
# 响应模型,schemes
# ============================================================

class UploadResponse(BaseModel):
    """上传响应"""
    success: List[dict] = Field(default_factory=list, description="成功列表")
    failed: List[dict] = Field(default_factory=list, description="失败列表")
    total_chunks: int = Field(0, description="总chunk数")
    message: str = Field(..., description="响应消息")


class SystemStatus(BaseModel):
    """系统状态"""
    storage_files: int = Field(..., description="存储文件数")
    rag_ready: bool = Field(..., description="RAG是否就绪")
    indexed_sources: List[str] = Field(default_factory=list, description="已索引文档")


# ============================================================
# 依赖注入
# ============================================================

def get_upload_service() -> UploadService:
    """获取UploadService实例"""
    return get_container().upload_service()


# ============================================================
# API端点
# ============================================================

@router.post("", response_model=UploadResponse, include_in_schema=False)
@router.post("/", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_service: UploadService = Depends(get_upload_service)
) -> UploadResponse:
    """
    上传文件并构建索引

    Args:
        files: 上传的文件列表
        upload_service: 上传服务

    Returns:
        上传结果
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="未提供文件")

        logger.info(f"[UploadAPI] 收到 {len(files)} 个文件")

        # 调用业务层
        results = await upload_service.upload_and_index(files)

        success_count = len(results["success"])
        failed_count = len(results["failed"])

        message = f"上传完成: 成功 {success_count} 个, 失败 {failed_count} 个"

        return UploadResponse(
            success=results["success"],
            failed=results["failed"],
            total_chunks=results["total_chunks"],
            message=message
        )

    except Exception as e:
        logger.error(f"[UploadAPI] 上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.get("/status", response_model=SystemStatus)
async def get_status(
    upload_service: UploadService = Depends(get_upload_service)
) -> SystemStatus:
    """
    获取系统状态

    Args:
        upload_service: 上传服务

    Returns:
        系统状态
    """
    try:
        status = upload_service.get_status()
        return SystemStatus(**status)

    except Exception as e:
        logger.error(f"[UploadAPI] 获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.delete("/clear")
async def clear_all(
    upload_service: UploadService = Depends(get_upload_service)
) -> dict:
    """
    清空所有文件和索引

    Args:
        upload_service: 上传服务

    Returns:
        操作结果
    """
    try:
        success = upload_service.clear_all()

        if success:
            return {"message": "已清空所有文件和索引"}
        else:
            raise HTTPException(status_code=500, detail="清空失败")

    except Exception as e:
        logger.error(f"[UploadAPI] 清空失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")
