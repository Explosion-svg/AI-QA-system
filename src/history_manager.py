"""
history_manager.py —— 聊天记录管理器
========================================
职责：把每一轮对话保存成 JSON 文件，下次打开程序可以继续上次的对话。

设计思路：
- 每个「会话」是一个独立的 JSON 文件，文件名 = session_id（时间戳）
- 文件存放在 chat_history/ 目录下
- 支持「增量保存」：每次 AI 回复后立刻写入磁盘，防止程序崩溃丢失记录

消息格式（与 OpenAI 接口保持一致）：
[
    {"role": "user",      "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你？"},
    ...
]
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict
from config import CHAT_SAVE_DIR, MAX_HISTORY


class HistoryManager:
    """
    聊天记录管理器。
    同时服务于 CLI（命令行）和 Streamlit 前端。
    """

    def __init__(self, save_dir: str = CHAT_SAVE_DIR, max_history: int = MAX_HISTORY):
        """
        :param save_dir:    聊天记录保存目录，默认 chat_history/
        :param max_history: 内存中最多保留多少轮对话（防止 token 超限）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)  # 目录不存在则自动创建
        self.max_history = max_history
        self._current: List[Dict] = []   # 当前会话的消息列表（内存中）

    # ==============================================================
    # 当前会话操作（内存层面）
    # ==============================================================

    def add(self, user: str, assistant: str):
        """
        添加一轮对话（用户问 + AI 答）到内存。
        如果超过最大历史轮数，自动丢弃最旧的记录（滚动窗口）。

        示例：
            mgr.add("今天天气怎样？", "今天晴天，气温 25℃。")
        """
        self._current.append({"role": "user",      "content": user})
        self._current.append({"role": "assistant", "content": assistant})

        # 超出上限时，从最旧的消息开始删除（每轮 = 2 条消息）
        if len(self._current) > self.max_history * 2:
            self._current = self._current[-(self.max_history * 2):]

    def get_history(self) -> List[Dict]:
        """
        返回当前内存中的完整对话历史列表（副本）。
        传给 LLMClient.chat() 作为上下文。
        """
        return list(self._current)

    def clear(self):
        """清空内存中的当前对话历史（不删除磁盘文件）。"""
        self._current = []

    # ==============================================================
    # 持久化操作（磁盘层面）
    # ==============================================================

    def save(self, session_id: str, messages: List[Dict], meta: dict = None) -> Path:
        """
        将对话记录保存为 JSON 文件。

        保存的 JSON 结构：
        {
            "session_id": "session_20240101_120000",
            "saved_at":   "2024-01-01 12:00:00",
            "meta":       {"provider": "deepseek", "model": "deepseek-chat"},
            "messages":   [{"role": "user", "content": "..."}, ...]
        }

        :param session_id: 会话唯一标识，通常是时间戳字符串
        :param messages:   要保存的消息列表
        :param meta:       附加信息，如使用的模型名称
        :return:           保存文件的路径
        """
        data = {
            "session_id": session_id,
            "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta":       meta or {},
            "messages":   messages,
        }
        path = self.save_dir / f"{session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def save_session(self, messages: List[Dict], meta: dict = None) -> Path:
        """
        Streamlit 专用：用「当天日期」作为 session_id 自动保存。
        同一天的对话会覆盖写入同一个文件。
        """
        session_id = time.strftime("session_%Y%m%d")
        return self.save(session_id, messages, meta)

    def load(self, session_id: str) -> List[Dict]:
        """
        从磁盘加载指定会话的消息列表。
        如果文件不存在，返回空列表（不报错）。

        示例：
            messages = mgr.load("session_20240101_120000")
        """
        path = self.save_dir / f"{session_id}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])

    def list_sessions(self) -> List[str]:
        """
        返回所有已保存会话的 ID 列表，按修改时间倒序（最新的在前）。
        用于 Streamlit 侧边栏的下拉选择框 和 CLI 的 /sessions 命令。
        """
        return [
            p.stem   # 取文件名（不含 .json 后缀）作为 session_id
            for p in sorted(
                self.save_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,   # 最新的排最前
            )
        ]

    def delete(self, session_id: str) -> bool:
        """
        删除指定会话的记录文件。
        :return: True 表示删除成功，False 表示文件不存在
        """
        path = self.save_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()   # unlink() 是 pathlib 的删除文件方法
            return True
        return False

    @staticmethod
    def new_session_id() -> str:
        """
        生成一个新的会话 ID（格式：session_年月日_时分秒）。
        静态方法：不需要实例化对象就能调用。

        示例返回值："session_20240101_153045"
        """
        return time.strftime("session_%Y%m%d_%H%M%S")
