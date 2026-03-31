"""
app.py —— Streamlit 前端界面
==============================
Streamlit 是一个用纯 Python 写网页应用的框架，无需学 HTML/CSS/JS。
每次用户操作，整个脚本从头到尾重新执行一次（这是 Streamlit 的核心机制）。
所以「需要跨次执行保留的数据」都要存在 st.session_state 里。

运行方式：
  streamlit run app.py
然后浏览器访问 http://localhost:8501

页面布局：
  左侧侧边栏 (st.sidebar)  —— 模型选择、RAG 设置、历史会话
  右侧主区域                —— 聊天气泡、输入框
"""

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（必须在 import config 之前调用）
load_dotenv()

from config import PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL, MAX_HISTORY
from llm_client import LLMClient
from history_manager import HistoryManager
from rag_engine import RAGEngine

# ==============================================================
# 页面基础配置（必须是第一个 Streamlit 调用）
# ==============================================================
st.set_page_config(
    page_title="AI 问答系统",
    page_icon="🤖",
    layout="wide",                  # 宽屏布局
    initial_sidebar_state="expanded",  # 侧边栏默认展开
)

# ==============================================================
# 自定义 CSS 样式（让聊天气泡更好看）
# ==============================================================
st.markdown("""
<style>
/* 用户消息气泡：蓝色背景，右对齐感 */
.chat-user {
    background: #e8f4fd;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    border-left: 3px solid #1890ff;
}
/* AI 消息气泡：灰色背景 */
.chat-ai {
    background: #f5f5f5;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    border-left: 3px solid #52c41a;
}
/* RAG 来源引用框：黄色边框 */
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
# Session State 初始化
# session_state 相当于「全局变量」，在用户刷新前一直保留
# 用 if xxx not in st.session_state 做「只初始化一次」的判断
# ==============================================================
def init_session():
    """初始化所有需要跨次执行保留的状态变量。"""
    if "messages" not in st.session_state:
        # 聊天记录列表，格式：[{role, content, sources?}, ...]
        st.session_state.messages = []

    if "provider" not in st.session_state:
        st.session_state.provider = DEFAULT_PROVIDER

    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    if "llm_client" not in st.session_state:
        # LLM 客户端，切换模型时重新创建
        st.session_state.llm_client = LLMClient(
            provider=st.session_state.provider,
            model=st.session_state.model,
        )

    if "history_mgr" not in st.session_state:
        st.session_state.history_mgr = HistoryManager(max_history=MAX_HISTORY)

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None   # 延迟初始化，用户开启 RAG 时才创建

    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False

    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False   # 知识库是否已就绪

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "你是一个智能助手，请用简洁准确的中文回答用户的问题。"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 2048


init_session()


# ==============================================================
# 侧边栏
# ==============================================================
with st.sidebar:
    st.title("⚙️ 设置")

    # ---- 模型选择 ----
    st.subheader("🤖 模型配置")

    # selectbox：下拉选择框
    # index 参数指定默认选中项
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

    # 服务商或模型变化时，重新创建 LLM 客户端
    if provider != st.session_state.provider or model != st.session_state.model:
        st.session_state.provider = provider
        st.session_state.model = model
        st.session_state.llm_client = LLMClient(provider=provider, model=model)
        st.success(f"已切换至 {PROVIDERS[provider]['icon']} {provider} / {model}")

    # API Key 可用性提示
    if not st.session_state.llm_client.is_available():
        st.warning(f"⚠️ {provider} 未配置有效的 API Key，请检查 .env 文件。")

    # 自定义系统提示词
    with st.expander("自定义系统提示词"):
        st.session_state.system_prompt = st.text_area(
            "系统提示词（定义 AI 角色）",
            value=st.session_state.system_prompt,
            height=100,
        )

    st.divider()

    # ---- RAG 知识库 ----
    st.subheader("📚 知识库 (RAG)")

    # toggle：开关按钮
    use_rag = st.toggle("启用知识库问答", value=st.session_state.use_rag)
    st.session_state.use_rag = use_rag

    if use_rag:
        # 文件上传组件，支持多文件
        uploaded_files = st.file_uploader(
            "上传文档（支持 txt / pdf / md）",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
            help="上传后点击【构建知识库】按钮生成向量索引",
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("构建知识库", type="primary"):
                if uploaded_files:
                    with st.spinner("正在构建向量索引，首次运行需下载 Embedding 模型（约 120MB）..."):
                        engine = RAGEngine()
                        count = engine.build_from_uploads(uploaded_files)
                        st.session_state.rag_engine = engine
                        st.session_state.rag_ready = True
                    st.success(f"构建完成，共 {count} 个文档块")
                else:
                    # 尝试加载已有索引
                    engine = RAGEngine()
                    if engine.load_index():
                        st.session_state.rag_engine = engine
                        st.session_state.rag_ready = True
                        st.success("已加载已有知识库索引")
                    else:
                        st.warning("请先上传文档")

        with col2:
            if st.button("清空索引"):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_index()
                    st.session_state.rag_engine = None
                    st.session_state.rag_ready = False
                    st.success("索引已清空")

        # 显示知识库状态
        if st.session_state.rag_ready and st.session_state.rag_engine:
            sources = st.session_state.rag_engine.list_sources()
            if sources:
                st.caption(f"已索引文件（{len(sources)} 个）：" + "、".join(sources[:5]))

    st.divider()

    # ---- 历史会话管理 ----
    st.subheader("🗂️ 历史会话")

    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.session_state.history_mgr.clear()
        st.rerun()   # 重新运行脚本，刷新界面

    # 列出历史会话文件，供用户选择加载
    session_list = st.session_state.history_mgr.list_sessions()
    if session_list:
        selected_session = st.selectbox(
            "加载历史会话",
            options=[""] + session_list,
            format_func=lambda x: "请选择..." if x == "" else x,
        )
        if selected_session and st.button("加载会话"):
            loaded_msgs = st.session_state.history_mgr.load(selected_session)
            # 转换为前端显示格式（兼容旧格式）
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                for m in loaded_msgs
            ]
            st.rerun()

    # 生成参数说明
    st.divider()
    with st.expander("生成参数"):
        st.session_state.temperature = st.slider(
            "Temperature（创造性）", 0.0, 1.0,
            st.session_state.temperature, 0.05,
            help="越低越保守严谨，越高越发散创意"
        )
        st.session_state.max_tokens = st.slider(
            "最大回复长度（tokens）", 256, 4096,
            st.session_state.max_tokens, 256
        )


# ==============================================================
# 主聊天区域
# ==============================================================
st.title("🤖 AI 问答系统")

# 状态栏：显示当前配置
col_a, col_b, col_c = st.columns(3)
col_a.metric("服务商", f"{PROVIDERS[st.session_state.provider]['icon']} {st.session_state.provider}")
col_b.metric("模型", st.session_state.model)
col_c.metric("知识库", "开启" if st.session_state.use_rag else "关闭")

st.divider()

# ---- 渲染历史消息 ----
# 遍历 session_state.messages，把每条消息渲染成气泡
for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])  # RAG 来源（可选）

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
        # 若有 RAG 来源，显示引用框
        if sources:
            src_html = "<br>".join(f"• {s}" for s in sources)
            st.markdown(
                f'<div class="source-box">📄 参考来源：<br>{src_html}</div>',
                unsafe_allow_html=True,
            )

# ---- 用户输入框 ----
# st.chat_input 会固定在页面底部，用户按 Enter 发送
user_input = st.chat_input("请输入您的问题…")

if user_input:
    # 1. 立即显示用户消息
    st.markdown(
        f'<div class="chat-user">🧑 <b>你</b><br>{user_input}</div>',
        unsafe_allow_html=True,
    )

    # 2. RAG 检索
    rag_context = ""
    sources = []
    if (
        st.session_state.use_rag
        and st.session_state.rag_ready
        and st.session_state.rag_engine
    ):
        rag_context, sources = st.session_state.rag_engine.get_context_with_sources(user_input)
        # rag_context = st.session_state.rag_engine.get_context(user_input)
        # st.write(f"[调试] RAG检索结果长度：{len(rag_context)}")  #调试用
        # 提取来源文件名供展示
        # results = st.session_state.rag_engine.retrieve(user_input)
        # sources = []
        # for doc, score in results:
        #     src = doc.metadata.get("source") or doc.metadata.get("file_path") or ""
        #     name = Path(src).name if src else "未知来源"
        #     page = doc.metadata.get("page", "")
        #     page_info = f" 第{page + 1}页" if page != "" else ""
        #     entry = f"{name}{page_info}（相似度 {score:.2f}）"
        #     if entry not in sources:
        #         sources.append(entry)

    # 3. 调用 LLM 获取回答
    with st.spinner("AI 思考中…"):
        # 获取历史消息（不含 sources 字段，只传 role/content 给 API）
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        try:
            answer = st.session_state.llm_client.chat(
                user_message=user_input,
                history=history,
                rag_context=rag_context,
                system_prompt=st.session_state.system_prompt,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
        except Exception as e:
            answer = f"请求出错: {e}\n\n请检查 API Key 配置或网络连接。"

    # 4. 显示 AI 回答
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

    # 5. 更新 session_state（保存本轮对话）
    st.session_state.messages.append({"role": "user",      "content": user_input, "sources": []})
    st.session_state.messages.append({"role": "assistant", "content": answer,     "sources": sources})

    # 6. 更新内存历史（用于下次调用时传给 API）
    st.session_state.history_mgr.add(user_input, answer)

    # 7. 自动保存会话到磁盘
    st.session_state.history_mgr.save_session(st.session_state.messages)
