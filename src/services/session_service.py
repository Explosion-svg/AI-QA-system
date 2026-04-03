"""
session_service.py —— 会话管理服务
==================================
职责：封装历史会话的业务逻辑
"""

import logging
from typing import List, Dict

from src.utils.history_manager import HistoryManager

logger = logging.getLogger(__name__)


class SessionService:
    """会话管理服务类"""

    def __init__(self):
        self.history_mgr = HistoryManager()

    def save_session(self, messages: List[Dict], metadata: Dict = None) -> str:
        """保存会话"""
        session_id = HistoryManager.new_session_id()
        self.history_mgr.save(session_id, messages, metadata)
        logger.info(f"[SessionService] 会话已保存: {session_id}")
        return session_id

    def load_session(self, session_id: str) -> List[Dict]:
        """加载会话"""
        messages = self.history_mgr.load(session_id)
        logger.info(f"[SessionService] 会话已加载: {session_id}")
        return messages

    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return self.history_mgr.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        success = self.history_mgr.delete(session_id)
        if success:
            logger.info(f"[SessionService] 会话已删除: {session_id}")
        return success
