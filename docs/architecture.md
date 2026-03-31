# 项目架构说明

## 目录结构

```
program3/
├── src/                    # 核心源码
│   ├── app.py              # Streamlit 前端入口
│   ├── cli.py              # 命令行入口
│   ├── config.py           # 全局配置
│   ├── llm_client.py       # LLM API 客户端（支持多服务商）
│   ├── rag_engine.py       # RAG 知识库引擎
│   └── history_manager.py  # 聊天历史管理
├── docs/                   # 项目文档
├── .streamlit/             # Streamlit 配置
├── knowledge_base/         # 用户上传文档（本地，不提交 git）
├── chat_history/           # 聊天历史文件（本地，不提交 git）
├── vector_db/              # ChromaDB 向量数据库（本地，不提交 git）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── start.bat               # Windows 一键启动
├── start.sh                # Linux/Mac 一键启动
└── README.md
```

## 模块说明

### src/app.py
Streamlit 前端，负责页面渲染、用户交互、调用 LLM 和 RAG。

### src/llm_client.py
统一封装多服务商 LLM 调用（OpenAI、DeepSeek、Anthropic 等），屏蔽 API 差异。

### src/rag_engine.py
RAG 引擎：文档加载 → 切片 → Embedding（sentence-transformers）→ ChromaDB 向量存储 → 语义检索。

### src/history_manager.py
管理聊天上下文窗口和本地会话持久化（JSON 文件）。

### src/config.py
集中管理所有可配置项，读取 `.env` 环境变量。

## 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | Streamlit |
| LLM | OpenAI / DeepSeek / Anthropic（可扩展） |
| Embedding | sentence-transformers（本地运行） |
| 向量数据库 | ChromaDB |
| 文档解析 | LangChain（PDF/TXT/MD/DOCX） |
| 容器化 | Docker + docker-compose |
