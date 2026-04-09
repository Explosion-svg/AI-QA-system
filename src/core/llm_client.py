"""
llm_client.py —— 统一 AI 对话接口
====================================
核心思想：「适配器模式」
不同厂商（OpenAI、DeepSeek、Qwen、Ollama）的 API 格式略有差异，
但它们都兼容 OpenAI 的接口标准。
所以我们只需要一个 OpenAI 客户端，切换 base_url 和 api_key 即可切换服务商。
就像同一个遥控器，换个信号频率就能控制不同品牌的电视。
"""

from __future__ import annotations
import os
import logging
from typing import Generator, List, Dict

logger = logging.getLogger(__name__)

from src.config import PROVIDERS, get_api_key, get_base_url


class LLMClient:
    """
    LLM（大语言模型）客户端。
    封装了「流式输出」和「普通输出」两种调用方式。
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """
        初始化客户端。
        :param provider: 服务商代号，如 'openai' / 'deepseek' / 'qwen' / 'ollama'
        :param model:    使用的模型名称，如 'gpt-4o' / 'deepseek-chat'
        """
        self.provider = provider
        self.model = model
        self._client = self._build_client()   # 创建底层 HTTP 客户端

    def _build_client(self):
        """
        根据当前 provider 构建 OpenAI 客户端实例。
        切换 base_url 是关键：不同服务商的地址不同，但接口格式一样。
        """
        from openai import OpenAI

        api_key = get_api_key(self.provider) or "ollama"  # Ollama 不验证 Key
        base_url = get_base_url(self.provider)
        logger.info(f"构建 LLM 客户端: provider={self.provider}, base_url={base_url}")
        return OpenAI(api_key=api_key, base_url=base_url)

    def switch(self, provider: str, model: str):
        """
        运行时切换服务商和模型，无需重启程序。
        CLI 中输入 /switch deepseek deepseek-chat 就会调用这里。
        """
        self.provider = provider
        self.model = model
        self._client = self._build_client()   # 重新创建客户端

    # ------------------------------------------------------------------
    # 流式输出：像 ChatGPT 一样，AI 边生成边返回，不用等全部生成完
    # 技术原理：Server-Sent Events（SSE），服务端推送每个 token
    # ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
        temperature: float = 0.7,    # 创造性：0=严谨保守，1=发散创意
        max_tokens: int = 2048,      # 单次回复最大 token 数
    ) -> Generator[str, None, None]:
        """
        流式生成器：每次 yield 一小段文字（通常是一个词或标点）。
        调用方用 for chunk in client.chat_stream(...): 来逐块处理。

        messages 格式：
        [
            {"role": "user",      "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user",      "content": "今天天气怎么样？"},
        ]
        """
        # 在历史消息前面插入系统提示词（定义 AI 的角色和行为）
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # stream=True 开启流式模式
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        # 遍历每个数据块，提取文字内容并逐个 yield 出去
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content   # 每次返回一小段文字

    # ------------------------------------------------------------------
    # 普通输出（非流式）：等 AI 全部生成完再一次性返回
    # Streamlit 前端使用这种方式更简单
    # ------------------------------------------------------------------
    def chat(
        self,
        user_message: str,
        history: List[Dict[str, str]] = None,
        rag_context: str = "",       # RAG 检索到的参考资料（可选）
        system_prompt: str = "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        发送消息，等待完整回复后返回字符串。
        如果提供了 rag_context，会自动把参考资料拼接到问题前面，
        让 AI 优先根据你的文档来回答。
        """
        messages = list(history or [])   # 复制历史，避免修改原列表

        # 如果有 RAG 上下文，重新构造用户消息（加上参考资料）
        final_user = user_message
        if rag_context:
            final_user = (
                f"请根据以下参考资料回答问题，如资料不足可结合自身知识。\n\n"
                f"参考资料:\n{rag_context}\n\n"
                f"问题: {user_message}"
            )

        messages.append({"role": "user", "content": final_user})

        # 把流式输出的所有块拼接成一个完整字符串
        return "".join(
            self.chat_stream(messages, system_prompt, temperature, max_tokens)
        )

    def is_available(self) -> bool:
        """
        检查当前服务商是否已配置有效的 API Key。
        用于启动时给出警告提示。
        """
        if self.provider == "ollama":
            return True   # 本地模型不需要 Key，始终视为可用
        key = get_api_key(self.provider)
        return bool(key and "your-" not in key)   # 排除未替换的占位符

    def list_models(self) -> List[str]:
        """
        返回当前服务商支持的模型列表。
        Ollama 会尝试从本地服务读取已安装的模型。
        """
        if self.provider == "ollama":
            try:
                import httpx
                base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                resp = httpx.get(f"{base}/api/tags", timeout=3)
                return [m["name"] for m in resp.json().get("models", [])]
            except Exception:
                pass   # 连接失败则返回默认列表
        return PROVIDERS[self.provider]["models"]


if __name__ == "__main__":
    """
    测试 LLMClient
    运行：python -m src.llm_client
    """
    print("=" * 50)
    print("测试 LLMClient")
    print("=" * 50)

    # 测试初始化
    client = LLMClient(provider="openai", model="gpt-3.5-turbo")
    print(f"✅ 客户端初始化成功: {client.provider}/{client.model}")
    print(f"API Key 可用: {client.is_available()}")

    # 测试对话
    if client.is_available():
        print("\n[测试] 发送简单对话...")
        try:
            answer = client.chat(
                user_message="用一句话介绍 Python",
                history=[],
                max_tokens=50
            )
            print(f"回答: {answer}")
        except Exception as e:
            print(f"❌ 对话失败: {e}")
    else:
        print("⚠️ API Key 未配置，跳过对话测试")

    print("\n✅ 测试完成")
