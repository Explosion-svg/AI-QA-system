"""
api.py —— 兼容入口
==================
保留旧启动命令 `uvicorn api:app` 的兼容性，实际应用统一复用 `main.py`。
"""

from main import app

__all__ = ["app"]
