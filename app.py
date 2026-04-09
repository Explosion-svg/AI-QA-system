"""
app.py —— Streamlit 前端界面
==============================
前端只负责 UI 展示，所有业务逻辑通过 HTTP 调用 api.py 完成。

运行方式（需先启动 api.py）：
  uvicorn api:app --host 0.0.0.0 --port 8000
  streamlit run app.py

页面布局：
  左侧侧边栏 (st.sidebar)  —— 模型选择、RAG 设置、历史会话
  右侧主区域                —— 聊天气泡、输入框
"""

import streamlit as st
import httpx
from dotenv import load_dotenv

load_dotenv()

from src.config import PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL, MAX_HISTORY
from src.memory.history_manager import HistoryManager

# API 服务地址
API_BASE = "http://127.0.0.1:8000"

# ==============================================================
# 页面基础配置
# ==============================================================
st.set_page_config(
    page_title="AI 问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.chat-user {
    background: #e8f4fd;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    border-left: 3px solid #1890ff;
}
.chat-ai {
    background: #f5f5f5;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    border-left: 3px solid #52c41a;
}
.source-box {
    background: #fffbe6;
    border-left: 4px solid #faad14;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    margin-top: 4px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================
# API 调用封装
# ==============================================================

def api_chat(message: str, history: list, use_rag: bool,
             provider: str, model: str, system_prompt: str,
             temperature: float, max_tokens: int) -> tuple[str, list]:
    """调用 POST /chat/，返回 (answer, sources)"""
    payload = {
        "message": message,
        "history": history,
        "use_rag": use_rag,
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = httpx.post(f"{API_BASE}/chat/", json=payload, timeout=120)
    resp.raise_for_status()     # 检查请求是否成功
    data = resp.json()
    return data["answer"], data.get("sources", [])


def api_upload_files(files) -> dict:
    """调用 POST /upload，返回响应 dict"""
    # python上传多文件，专门给requests库使用
    files_payload = [
        ("files", (f.name, f.read(), "application/octet-stream"))
        for f in files
    ]
    resp = httpx.post(f"{API_BASE}/upload/", files=files_payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def api_kb_status() -> dict:
    """调用 GET /chat/status，返回状态 dict"""
    try:
        resp = httpx.get(f"{API_BASE}/chat/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"ready": False, "sources": [], "source_count": 0, "message": "API 未连接"}


def api_clear_kb() -> bool:
    """调用 DELETE /upload/clear，返回是否成功"""
    resp = httpx.delete(f"{API_BASE}/upload/clear", timeout=30)
    resp.raise_for_status()
    return True


def api_health() -> bool:
    """检查 API 服务是否在线"""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ==============================================================
# Session State 初始化
# ==============================================================
def init_session():
    defaults = {
        "messages": [],
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "use_rag": False,
        "system_prompt": "你是一个智能助手，请用简洁准确的中文回答用户的问题。",
        "temperature": 0.7,
        "max_tokens": 2048,
        "history_mgr": HistoryManager(max_history=MAX_HISTORY),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ==============================================================
# 侧边栏
# ==============================================================
with st.sidebar:
    st.title("⚙️ 设置")

    # API 健康状态
    api_online = api_health()
    if api_online:
        st.success("✅ API 服务已连接", icon="🟢")
    else:
        st.error("❌ API 服务未连接，请先启动 api.py", icon="🔴")

    # 分割线
    st.divider()

    # ---- 模型选择 ----
    st.subheader("🤖 模型配置")

    provider = st.selectbox(
        "服务商",
        options=list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider),
        format_func=lambda k: f"{PROVIDERS[k]['icon']} {PROVIDERS[k]['name']}",
    )
    model = st.selectbox(
        "模型",
        options=PROVIDERS[provider]["models"],
    )
    if provider != st.session_state.provider or model != st.session_state.model:
        st.session_state.provider = provider
        st.session_state.model = model
        st.success(f"已切换至 {PROVIDERS[provider]['icon']} {provider} / {model}")

    with st.expander("自定义系统提示词"):
        st.session_state.system_prompt = st.text_area(
            "系统提示词（定义 AI 角色）",
            value=st.session_state.system_prompt,
            height=100,
        )

    st.divider()

    # ---- RAG 知识库 ----
    st.subheader("📚 知识库 (RAG)")

    use_rag = st.toggle("启用知识库问答", value=st.session_state.use_rag)
    st.session_state.use_rag = use_rag

    if use_rag:
        uploaded_files = st.file_uploader(
            "上传文档（支持 txt / pdf / md / docx）",
            type=["txt", "pdf", "md", "docx"],
            accept_multiple_files=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("构建知识库", type="primary", disabled=not api_online):
                if uploaded_files:
                    with st.spinner("正在上传并构建向量索引..."):
                        try:
                            result = api_upload_files(uploaded_files)
                            if result.get("success"):
                                st.success(result.get("message", "构建成功"))
                            else:
                                st.error(result.get("message", "构建失败"))
                        except Exception as e:
                            st.error(f"上传失败: {e}")
                else:
                    st.warning("请先上传文档")

        with col2:
            if st.button("清空索引", disabled=not api_online):
                try:
                    if api_clear_kb():
                        st.success("索引已清空")
                    else:
                        st.warning("清空失败或索引不存在")
                except Exception as e:
                    st.error(f"清空失败: {e}")

        # 显示知识库状态（实时从 API 获取）
        if use_rag and api_online:
            kb_status = api_kb_status()
            if kb_status.get("ready"):
                sources = kb_status.get("sources", [])
                st.caption(
                    f"已索引文件（{len(sources)} 个）：" + "、".join(sources[:5])
                    if sources else "知识库已就绪"
                )
            else:
                st.caption("⚠️ 知识库未构建，请上传文档")

    st.divider()

    # ---- 历史会话管理 ----
    st.subheader("🗂️ 历史会话")

    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.session_state.history_mgr.clear()
        st.rerun()

    session_list = st.session_state.history_mgr.list_sessions()
    if session_list:
        selected_session = st.selectbox(
            "加载历史会话",
            options=[""] + session_list,
            format_func=lambda x: "请选择..." if x == "" else x,
        )
        if selected_session and st.button("加载会话"):
            loaded_msgs = st.session_state.history_mgr.load(selected_session)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                for m in loaded_msgs
            ]
            st.rerun()

    st.divider()
    with st.expander("生成参数"):
        st.session_state.temperature = st.slider(
            "Temperature（创造性）", 0.0, 1.0,
            st.session_state.temperature, 0.05,
        )
        st.session_state.max_tokens = st.slider(
            "最大回复长度（tokens）", 256, 4096,
            st.session_state.max_tokens, 256,
        )


# ==============================================================
# 主聊天区域
# ==============================================================
st.title("🤖 AI 问答系统")

col_a, col_b, col_c = st.columns(3)
col_a.metric("服务商", f"{PROVIDERS[st.session_state.provider]['icon']} {st.session_state.provider}")
col_b.metric("模型", st.session_state.model)
col_c.metric("知识库", "开启" if st.session_state.use_rag else "关闭")

st.divider()

# ---- 渲染历史消息 ----
for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f'<div class="chat-user">🧑 <b>你</b><br>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-ai">🤖 <b>AI</b><br>{content}</div>',
            unsafe_allow_html=True,
        )
        if sources:
            src_html = "<br>".join(f"• {s}" for s in sources)
            st.markdown(
                f'<div class="source-box">📄 参考来源：<br>{src_html}</div>',
                unsafe_allow_html=True,
            )

# ---- 用户输入框 ----
user_input = st.chat_input("请输入您的问题…", disabled=not api_online)

if user_input:
    st.markdown(
        f'<div class="chat-user">🧑 <b>你</b><br>{user_input}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("AI 思考中…"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        try:
            answer, sources = api_chat(
                message=user_input,
                history=history,
                use_rag=st.session_state.use_rag,
                provider=st.session_state.provider,
                model=st.session_state.model,
                system_prompt=st.session_state.system_prompt,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
        except Exception as e:
            answer = f"请求出错: {e}\n\n请检查 API 服务是否正常运行。"
            sources = []

    st.markdown(
        f'<div class="chat-ai">🤖 <b>AI</b><br>{answer}</div>',
        unsafe_allow_html=True,
    )
    if sources:
        src_html = "<br>".join(f"• {s}" for s in sources)
        st.markdown(
            f'<div class="source-box">📄 参考来源：<br>{src_html}</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({"role": "user",      "content": user_input, "sources": []})
    st.session_state.messages.append({"role": "assistant", "content": answer,      "sources": sources})

    st.session_state.history_mgr.add(user_input, answer)
    st.session_state.history_mgr.save_session(st.session_state.messages)
