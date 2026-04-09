# RAG系统架构文档

## 目录结构

```
project3/
│
├── main.py                     # FastAPI启动入口
│
├── src/
│   │
│   ├── api/                    # API层（HTTP接口）
│   │   ├── __init__.py
│   │   ├── chat_api.py         # 聊天接口
│   │   ├── upload_api.py       # 上传接口
│   │   └── schemas.py          # 数据模型
│   │
│   ├── services/               # 业务层
│   │   ├── __init__.py
│   │   ├── chat_service.py     # 聊天业务编排
│   │   └── upload_service.py   # 上传业务编排
│   │
│   ├── rag/                    # RAG能力层
│   │   ├── __init__.py
│   │   ├── rag_engine.py       # RAG流程编排
│   │   ├── rewrite.py          # 查询改写
│   │   ├── retriever.py        # 混合检索
│   │   ├── rerank.py           # 重排序
│   │   ├── context_filter.py   # 上下文过滤
│   │   └── prompt_builder.py   # Prompt构建
│   │
│   ├── index/                  # 向量索引层
│   │   ├── __init__.py
│   │   ├── vector_store.py     # 抽象接口
│   │   ├── chroma_store.py     # ChromaDB实现
│   │   └── document_loader.py  # 文档加载器
│   │
│   ├── memory/                 # 记忆层
│   │   ├── __init__.py
│   │   └── history_manager.py  # 历史管理
│   │
│   ├── infra/                  # 基础设施层
│   │   ├── __init__.py
│   │   ├── llm_client.py       # LLM客户端
│   │   ├── embedding_model.py  # Embedding模型
│   │   ├── config.py           # 配置管理
│   │   └── logger.py           # 日志管理
│   │
│   ├── container.py            # 依赖注入容器
│   └── cli.py                  # 命令行工具
│
├── requirements.txt
├── .env
└── README.md
```

## 分层职责

### 1. main.py（系统入口）
- 创建FastAPI应用
- 初始化容器
- 注册路由
- 配置生命周期

### 2. container.py（依赖注入）
- 创建所有核心对象
- 管理对象生命周期
- 连接依赖关系

### 3. API层
- 接收HTTP请求
- 参数校验
- 调用Service
- 返回响应

### 4. Services层
- 业务流程编排
- 调用RAG引擎
- 调用历史管理
- 不涉及HTTP细节

### 5. RAG层
- 实现RAG pipeline
- 查询改写
- 混合检索
- 重排序
- 上下文过滤

### 6. Index层
- 向量数据库操作
- 文档加载
- Chunk切分
- Embedding

### 7. Memory层
- 对话历史管理
- 加载/保存历史
- 历史截断

### 8. Infra层
- LLM API调用
- Embedding模型
- 配置管理
- 日志管理

## RAG检索流程

```
用户查询 (query)
    ↓
[Query Rewriter] 查询改写
    ↓
[Hybrid Retriever] 混合检索
    ├─→ BM25检索（关键词匹配）
    └─→ 向量检索（语义匹配）
    ↓
[RRF Fusion] 融合两路结果
    ↓
[Reranker] CrossEncoder重排序
    ↓
[Context Filter] 上下文过滤
    ↓
返回最终上下文 → LLM生成
```

## 核心设计原则

1. **分层隔离**：各层职责单一，互不穿透
2. **依赖注入**：通过Container管理对象创建
3. **接口抽象**：VectorStore等使用抽象接口
4. **生命周期管理**：统一的startup/shutdown
5. **配置集中**：所有配置在infra/config.py

## 启动方式

### 1. Web API
```bash
python3 main.py
```

### 2. 命令行
```bash
python3 -m src.cli chat
```

## 环境变量

在`.env`文件中配置：

```env
# LLM配置
OPENAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo

# RAG配置
CHUNK_SIZE=
CHUNK_OVERLAP=
RAG_TOP_K=

# 路径配置
KNOWLEDGE_DIR=knowledge_base
CHAT_SAVE_DIR=chat_history
```


