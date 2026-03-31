# AI 问答系统

支持 OpenAI / DeepSeek / Qwen / Ollama 的全功能 AI 问答系统，具备流式输出、RAG 知识库、多轮对话、聊天记录保存等功能。

---

## 项目结构

```
program3/
├── .env.example        # API Key 配置模板
├── requirements.txt    # 依赖列表
├── config.py           # 全局配置
├── llm_client.py       # 统一 LLM 客户端（流式 + 非流式）
├── history_manager.py  # 聊天记录自动保存与加载
├── rag_engine.py       # RAG 知识库引擎
├── cli.py              # 命令行工具
├── app.py              # Streamlit 前端
├── chat_history/       # 自动保存的对话记录（运行后生成）
├── knowledge_base/     # 知识库文档目录（运行后生成）
└── vector_db/          # 向量数据库（构建知识库后生成）
```

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制配置模板并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，按需填写对应服务商的 API Key：

```ini
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# DeepSeek
DEEPSEEK_API_KEY=your-deepseek-api-key

# 通义千问 (Qwen)
QWEN_API_KEY=your-qwen-api-key

# 本地 Ollama（无需 Key，保持默认即可）
OLLAMA_BASE_URL=http://localhost:11434
```

> 只需填写你实际使用的服务商，其余可留空。

### 3. 启动方式

#### 方式一：Streamlit 前端（推荐）

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，在侧边栏选择服务商和模型，即可开始对话。

#### 方式二：命令行工具

```bash
# 使用 DeepSeek
python cli.py chat --provider deepseek --model deepseek-chat

# 使用 OpenAI GPT-4
python cli.py chat --provider openai --model gpt-4

# 使用本地 Ollama 模型
python cli.py chat --provider ollama --model qwen2

# 启用 RAG 知识库
python cli.py chat --provider deepseek --model deepseek-chat --rag

# 加载历史会话
python cli.py chat --session session_20240101_120000
```

---

## 功能说明

### 支持的模型服务商

| 服务商 | 标识 | 代表模型 | 是否需要 Key |
|--------|------|----------|--------------|
| OpenAI | `openai` | gpt-3.5-turbo, gpt-4o | 是 |
| DeepSeek | `deepseek` | deepseek-chat, deepseek-reasoner | 是 |
| 通义千问 | `qwen` | qwen-turbo, qwen-plus, qwen-max | 是 |
| Ollama 本地 | `ollama` | llama3, qwen2, mistral, phi3 | 否 |

### 流式输出

命令行模式下 AI 回复实时逐字打印，效果类似 ChatGPT。

### 多轮对话

自动维护对话上下文，支持连续追问。历史长度可在 `.env` 中通过 `MAX_HISTORY` 配置。

### 自动保存聊天记录

每次 AI 回复后自动保存到 `chat_history/` 目录（JSON 格式），下次可直接加载继续对话。

### RAG 知识库

将文档放入 `knowledge_base/` 目录（或通过 Streamlit 上传），系统自动切片、向量化，回答时自动检索相关内容作为参考。

支持格式：`.txt` / `.md` / `.pdf` / `.docx`

**命令行 RAG 命令：**

| 命令 | 说明 |
|------|------|
| `/rag on` | 开启知识库检索 |
| `/rag off` | 关闭知识库检索 |
| `/rag build` | 重新构建向量索引 |
| `/rag add <文件路径>` | 增量添加文档 |

### 命令行内置命令

| 命令 | 说明 |
|------|------|
| `/switch <provider> <model>` | 切换模型，如 `/switch qwen qwen-max` |
| `/history` | 查看当前对话历史 |
| `/sessions` | 查看所有历史会话 |
| `/load <session_id>` | 加载指定历史会话 |
| `/new` | 开启新会话 |
| `/clear` | 清空当前对话历史 |
| `/models` | 列出所有支持的模型 |
| `/status` | 查看当前配置状态 |
| `/help` | 显示帮助信息 |
| `/exit` | 保存并退出 |

---

## 本地模型（Ollama）

无需 API Key，完全本地运行，保护隐私。

### 安装 Ollama

前往 [https://ollama.com](https://ollama.com) 下载并安装。

### 下载模型

```bash
# 推荐轻量模型
ollama pull qwen2          # 通义千问 2（中文友好）
ollama pull llama3:8b      # Meta LLaMA 3 8B
ollama pull phi3           # 微软 Phi-3（极轻量）
ollama pull mistral        # Mistral 7B
```

### 启动对话

```bash
python cli.py chat --provider ollama --model qwen2
```

---

## 常见问题

**Q: 提示 API Key 无效怎么办？**
A: 检查 `.env` 文件是否存在，Key 是否正确填写，注意不要有多余空格。

**Q: Ollama 连接失败？**
A: 确保 Ollama 服务已启动（运行 `ollama serve`），默认端口为 11434。

**Q: 知识库检索结果不准确？**
A: 尝试 `/rag build` 重建索引，或调整 `config.py` 中的 `CHUNK_SIZE` 和 `RAG_TOP_K` 参数。

**Q: 如何更换 Embedding 模型？**
A: 修改 `config.py` 中的 `EMBEDDING_MODEL`，首次运行会自动下载。

---

## 依赖说明

| 依赖 | 用途 |
|------|------|
| openai | 统一调用各服务商 API |
| python-dotenv | 读取 .env 配置 |
| streamlit | Web 前端界面 |
| langchain | RAG 文档处理流水线 |
| chromadb | 本地向量数据库 |
| sentence-transformers | 本地 Embedding 模型 |
| rich / typer | 命令行美化与参数解析 |
| pypdf / python-docx | PDF / Word 文档解析 |
