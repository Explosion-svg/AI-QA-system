# 项目重构说明

## 新的目录结构

```
src/
├── api/                    # API 层
│   ├── __init__.py
│   └── schemas.py         # Pydantic 数据模型
│
├── services/              # 业务逻辑层
│   ├── __init__.py
│   ├── chat_service.py    # 对话服务
│   ├── session_service.py # 会话管理服务
│   └── upload_service.py  # 文档上传服务
│
├── core/                  # 核心功能层
│   ├── __init__.py
│   ├── llm_client.py      # LLM 客户端
│   ├── rag_engine.py      # RAG 引擎
│   ├── parser.py          # 文档解析器
│   └── storage.py         # 文件存储管理器
│
├── utils/                 # 工具层
│   ├── __init__.py
│   └── history_manager.py # 历史记录管理
│
├── config.py              # 全局配置
└── cli.py                 # CLI 入口
```

## 分层说明

### 1. API 层 - 数据模型定义
### 2. 服务层 - 业务逻辑封装
### 3. 核心层 - 底层功能实现
### 4. 工具层 - 通用工具函数

## 优势

- 职责清晰，易于维护
- 模块独立，可单独测试
- 符合工业标准
