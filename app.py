"""Streamlit 前端：文档知识库问答 Agent 的可视化界面。

运行方式（在项目根目录 doc-rag-agent 下）:
    streamlit run app.py
"""

import os
import sys
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

# 确保能导入同目录下的 rag_agent 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_agent.agent import create_rag_agent  # noqa: E402
from rag_agent.tools import KnowledgeBase  # noqa: E402
from zipagent import Runner, StreamEventType  # noqa: E402

load_dotenv()

st.set_page_config(
    page_title="文档知识库问答 Agent", page_icon="📚", layout="wide"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _list_data_files() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith((".md", ".txt"))
    )


def _rebuild_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    paths = _list_data_files()
    if paths:
        kb.load(paths)
    return kb


def _render_run(agent, prompt: str):
    """运行 Agent，并把 思考/工具调用/检索结果/回答 实时渲染到界面。"""
    answer_ph = st.empty()
    status_box = st.container()
    full = ""

    def callback(event) -> None:
        nonlocal full
        if event.type == StreamEventType.THINKING and event.content:
            status_box.caption(f"💭 {event.content[:150]}")
        elif event.type == StreamEventType.TOOL_CALL:
            status_box.caption(f"🔧 调用工具：`{event.tool_name}`")
        elif event.type == StreamEventType.TOOL_RESULT:
            with status_box.expander("📊 检索到的片段", expanded=True):
                st.write(event.tool_result)
        elif event.type == StreamEventType.ANSWER_DELTA:
            full += event.content
            answer_ph.markdown(full)
        elif event.type == StreamEventType.ANSWER:
            answer_ph.markdown(event.content or full)
        elif event.type == StreamEventType.ERROR:
            answer_ph.error(event.error)

    result = Runner.run(agent, prompt, stream_callback=callback)
    return result


def main() -> None:
    st.title("📚 文档知识库问答 Agent（RAG）")
    st.caption("基于 ZipAgent + DeepSeek 的检索增强生成问答系统")

    # 会话状态初始化
    if "kb" not in st.session_state:
        st.session_state.kb = _rebuild_kb()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = None

    # 侧边栏：知识库管理
    with st.sidebar:
        st.header("🗂️ 知识库")
        kb = st.session_state.kb
        if kb.chunks:
            counts = Counter(c.source for c in kb.chunks)
            for src, n in counts.items():
                st.text(f"· {os.path.basename(src)}：{n} 片段")
        else:
            st.info("知识库为空")

        st.divider()
        uploaded = st.file_uploader(
            "上传文档（.md / .txt）", type=["md", "txt"], accept_multiple_files=True
        )
        if uploaded:
            os.makedirs(DATA_DIR, exist_ok=True)
            for f in uploaded:
                safe_name = os.path.basename(f.name)
                with open(os.path.join(DATA_DIR, safe_name), "wb") as out:
                    out.write(f.getbuffer())
            st.session_state.kb = _rebuild_kb()
            st.success(f"已上传 {len(uploaded)} 个文件，知识库已更新")

        if st.button("🔄 重新加载 data/ 目录"):
            st.session_state.kb = _rebuild_kb()
            st.rerun()

        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.context = None
            st.rerun()

        st.divider()
        st.caption("模型：DeepSeek deepseek-chat")

    # 聊天区
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if not os.getenv("API_KEY"):
        st.warning("未检测到 API_KEY，请在 `.env` 中配置后重启应用。")
        return

    if prompt := st.chat_input("问一个关于文档的问题…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        agent = create_rag_agent(st.session_state.kb)

        with st.chat_message("assistant"):
            result = _render_run(agent, prompt)

        st.session_state.context = result.context
        st.session_state.messages.append(
            {"role": "assistant", "content": result.content or ""}
        )


if __name__ == "__main__":
    main()
