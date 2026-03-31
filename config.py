"""
config.py —— 全局配置中心
============================
这个文件是整个项目的「大脑控制台」。
所有模型参数、路径、API 地址都在这里统一管理。
好处：改一个地方，全项目生效，不用到处找代码改。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（API Key 等敏感信息放在那里，不写死在代码里）
load_dotenv()


# ============================================================
# 模型提供商配置字典
# 每个 key 是提供商的「代号」，value 是它的详细信息
# 新增一个提供商，只需在这里加一项即可
# ============================================================
PROVIDERS = {
    "openai": {
        "name": "OpenAI",                              # 显示名称
        "api_key_env": "OPENAI_API_KEY",               # 从哪个环境变量读取 Key
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # 接口地址
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],         # 可用模型列表
        "icon": "🟢",                                   # 侧边栏显示的图标
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "icon": "🔵",
    },
    "qwen": {
        "name": "通义千问 (Qwen)",
        "api_key_env": "QWEN_API_KEY",
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
        "icon": "🟡",
    },
    "ollama": {
        # Ollama 是本地运行的模型框架，不需要 API Key，完全离线，保护隐私
        "name": "Ollama (本地)",
        "api_key_env": None,                            # None 表示不需要 Key
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
        "models": ["llama3", "llama3:8b", "mistral", "qwen2", "phi3", "gemma2"],
        "icon": "🏠",
    },
}

# ============================================================
# 应用默认值（可通过 .env 覆盖）
# ============================================================
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")   # 默认服务商
DEFAULT_MODEL    = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")  # 默认模型
MAX_HISTORY      = int(os.getenv("MAX_HISTORY", "20"))        # 最多保留多少轮对话历史
CHAT_SAVE_DIR    = os.getenv("CHAT_SAVE_DIR", "chat_history") # 聊天记录保存目录
KNOWLEDGE_DIR    = os.getenv("KNOWLEDGE_DIR", "knowledge_base")  # 知识库文档目录

# ============================================================
# RAG（检索增强生成）相关配置
# RAG 原理：把你的文档切成小块 -> 转成向量 -> 用户提问时找最相关的块 -> 喂给 AI
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 上面这个模型：多语言、支持中文、体积小（~120MB）、首次运行自动下载

CHUNK_SIZE    = 500   # 每个文档块的最大字符数（太大检索不精准，太小上下文不足）
CHUNK_OVERLAP = 50    # 相邻块之间的重叠字符数（保证切割处语义不断裂）
RAG_TOP_K     = 4     # 每次检索返回最相关的前 K 个文档块

# ============================================================
# 模型下载路径、加速
# ============================================================
# if os.getenv("HF_HOME"):
#     os.environ["HF_HOME"] = os.getenv("HF_HOME")
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")


# ============================================================
# 工具函数
# ============================================================

def get_api_key(provider: str) -> str | None:
    """
    根据提供商名称，从环境变量中读取对应的 API Key。
    例如：get_api_key("openai") 会读取环境变量 OPENAI_API_KEY 的值。
    """
    cfg = PROVIDERS.get(provider, {})
    env = cfg.get("api_key_env")
    if env is None:
        return "ollama"   # Ollama 不需要真实 Key，随便填一个字符串即可
    return os.getenv(env, "")


def get_base_url(provider: str) -> str:
    """返回指定提供商的 API 请求地址。"""
    return PROVIDERS[provider]["base_url"]


def list_providers() -> list:
    """返回所有支持的提供商代号列表，例如 ['openai', 'deepseek', 'qwen', 'ollama']。"""
    return list(PROVIDERS.keys())
