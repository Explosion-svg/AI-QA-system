# RAG知识库问答系统

基于RAG（检索增强生成）的智能问答系统，采用企业级分层架构设计。

## 特性

- 🏗️ **企业级架构**：清晰的分层设计，职责分离
- 🔍 **混合检索**：向量检索 + 重排序
- 🚀 **高性能**：依赖注入、生命周期管理
- 📚 **多格式支持**：PDF、DOCX、TXT
- 🔌 **多模型支持**：OpenAI、DeepSeek、Qwen、Ollama
- 💾 **对话记忆**：会话历史管理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建`.env`文件：

```env
OPENAI_API_KEY=your-key
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo
```

### 3. 启动服务

```bash
# Web API
python3 main.py

# 命令行
python3 -m src.cli chat
```

### 4. 访问API文档

打开浏览器访问：http://localhost:8000/docs

## 项目结构

```
project3/
├── main.py                 # FastAPI入口
├── src/
│   ├── api/               # HTTP接口层
│   ├── services/          # 业务逻辑层
│   ├── rag/               # RAG能力层
│   ├── index/             # 向量索引层
│   ├── memory/            # 记忆层
│   ├── infra/             # 基础设施层
│   └── container.py       # 依赖注入
└── requirements.txt
```

## API使用

### 上传文档

```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "files=@document.pdf"
```

### 聊天

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什么是RAG？",
    "use_rag": true
  }'
```

## 架构设计

详见 [docs/architecture.md](docs/architecture.md)

## 技术栈

- FastAPI - Web框架
- ChromaDB - 向量数据库
- LangChain - 文档处理
- sentence-transformers - Embedding和Reranker

## License

MIT
