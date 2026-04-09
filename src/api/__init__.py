"""
api —— HTTP接口层
==================
职责：接收HTTP请求、参数校验、调用service、返回响应
"""

from .chat_api import router as chat_router
from .upload_api import router as upload_router

__all__ = ["chat_router", "upload_router"]
