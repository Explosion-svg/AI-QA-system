"""
config.py —— 兼容配置入口
=========================
旧代码统一转发到 `src.infra.config` 和 `src.infra.logger`。
"""

import logging

from src.infra.config import *  # noqa: F401,F403
from src.infra.logger import setup_logger


def setup_logging(level: str = "INFO"):
    """兼容旧入口使用的 setup_logging 名称。"""
    return setup_logger(level=getattr(logging, level.upper(), logging.INFO), log_file="logs/app.log")
