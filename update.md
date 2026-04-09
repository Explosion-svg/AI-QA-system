# 一、项目架构说明

## 目录结构

```
project3/
│
├── src/
│   │
│   ├── api/					# API层（HTTP接口）
│   │   ├── __init__.py
│   │   ├── chat_api.py
│   │   ├── upload_api.py
│   │   └── schemas.py
│   │
│   ├── services/				# 业务层
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   └── upload_service.py
│   │
│   ├── rag/					# AI能力层
│   │   ├── __init__.py
│   │   ├── rag_engine.py
│   │   ├── rewrite.py
│   │   ├── retriever.py
│   │   ├── rerank.py
|	|	├──	context_filter.py
│   │   └── prompt_builder.py
│   │
│   ├── index/					# 向量索引层
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   ├── chroma_store.py
│   │   └── document_loader.py
│   │
│   ├── memory/					# 记忆层
│   │   ├── __init__.py
│   │   └── history_manager.py
│   │
│   ├── infra/					# 基础设施层
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── embedding_model.py
│   │   ├── logger.py
│   │   └── config.py
│   │
│   └── container.py			# ⭐系统依赖装配
│
└── main.py                 ⭐ FastAPI启动入口
```

# 二、Responsibility Boundary

## 1.main.py（系统入口）

### 职责

系统启动入口。

### 负责

```
创建FastAPI
初始化container
注册router
配置lifespan
```

### 不负责

```
业务逻辑
RAG调用
向量库操作
```

## 2.container.py（系统依赖装配）

### 职责

**系统对象创建中心（Dependency Injection）**

### 负责

```
创建所有核心对象
连接依赖关系
管理单例对象
```

### 不负责

```
业务逻辑
HTTP
RAG流程
```

包含startup()，shutdown()等方法

## 3.api 层（HTTP接口）

### 职责

HTTP请求入口。

### 负责

```
接收HTTP请求
参数校验
调用service
返回response
```

### 不负责

```
RAG逻辑
数据库
向量库
```

## chat_api.py

### 职责

```
处理聊天请求
```

## upload_api.py

### 职责

```
文件上传接口
```

## schemas.py

### 职责

```
API请求/响应数据结构
```

## 4.services 层（业务流程）

### 职责

业务逻辑编排。

### 负责

```
组织系统流程
调用RAG
调用memory
```

### 不负责

```
向量检索实现
LLM调用细节
HTTP
```

## chat_service.py

### 职责

```
聊天业务流程
```

### 流程

```
加载历史
调用RAG
保存历史
返回回答
```

## upload_service.py

### 职责

```
文件上传业务
```

### 流程

```
保存文件
加载文档
构建索引
```

# 5.rag 层（AI能力）

### 职责

实现 **RAG pipeline**

### 负责

```
rewrite
retrieve
rerank
prompt
generate
```

### 不负责

```
HTTP
用户
历史管理
```

## rag_engine.py

### 职责

```
RAG流程 orchestrator
```

## rewrite.py

### 职责

```
查询改写
```

## rerank.py

### 职责

```
文档重排序
```

### context_filter.py

### 职责

```
保持不变
```

## prompt_builder.py

### 职责

```
构建最终prompt
```

### 删除bm25_retriever.py和hybrid_fusion.py

# 6.index 层（向量索引）

### 职责

```
向量数据库操作
```

### 负责

```
文档加载
chunk切片
embedding
向量存储
向量检索
```

## vector_store.py

### 职责

```
抽象接口。
```

## chroma_store.py

### 职责

```
ChromaDB实现
```

## document_loader.py

### 职责

文档读取

```
pdf
txt
docx
```

# 7.memory 层

### 职责

对话记忆管理。

### history_manager.py

### 负责

```
加载历史
保存历史
截断历史
```

# 8.infra 层

### 职责

所有 **外部系统接口**。

## llm_client.py

负责

```
调用LLM API
```

## embedding_model.py

负责

```
文本 embedding
```

## logger.py

负责

```
日志封装
```

## config.py

负责

```
系统配置
```

# 三、核心设计原则

### 分层隔离

- 各层职责单一，互不穿透
