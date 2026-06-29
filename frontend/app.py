import streamlit as st
import requests
import uuid
import base64
import re

st.set_page_config(page_title="Data Agent — 智能数据分析", layout="wide", page_icon="📊")

with open("frontend/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# ===================== 会话状态 =====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_list" not in st.session_state:
    st.session_state.thread_list = []
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None


def decode_chart(chart_data: str):
    if not chart_data:
        return None
    if chart_data.startswith("data:"):
        b64 = re.sub(r"^data:image/\w+;base64,", "", chart_data)
    else:
        b64 = chart_data
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def load_thread_list():
    try:
        resp = requests.get(f"{API_URL}/threads", timeout=5)
        if resp.status_code == 200:
            st.session_state.thread_list = resp.json().get("threads", [])
    except Exception:
        pass


def load_history(thread_id: str):
    try:
        resp = requests.get(f"{API_URL}/history", params={"thread_id": thread_id}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            msgs = []
            for m in data.get("history", []):
                role = "user" if m["role"] == "human" else "assistant"
                msgs.append({"role": role, "content": m["content"]})
            st.session_state.messages = msgs
            st.session_state.thread_id = thread_id
    except Exception:
        pass


def delete_thread(thread_id: str):
    try:
        resp = requests.delete(f"{API_URL}/threads/{thread_id}", timeout=5)
        if resp.status_code == 200:
            if st.session_state.thread_id == thread_id:
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
            load_thread_list()
    except Exception:
        pass


# ===================== 侧边栏 =====================
with st.sidebar:
    st.markdown("## 📊 Data Agent")
    st.caption("智能数据分析助手")

    st.divider()

    if st.button("➕ 新对话", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.file_id = None
        st.session_state.delete_confirm = None

    st.divider()

    st.markdown("### 📁 数据上传")
    uploaded_file = st.file_uploader("支持 CSV / Excel / Word", type=["csv", "xlsx", "xls", "docx"],
                                     label_visibility="collapsed")
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("解析文件中..."):
            resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.file_id = data["file_id"]
            st.success(f"✅ {data['filename']} 已加载")
            st.caption(f"行数: {data['shape'][0]} · 列数: {data['shape'][1]}")
        else:
            st.error(f"上传失败: {resp.json().get('detail', '未知错误')}")

    st.divider()

    st.markdown("### 💬 历史会话")
    col_refresh, _ = st.columns([1, 2])
    with col_refresh:
        if st.button("🔄 刷新", use_container_width=True):
            load_thread_list()

    if not st.session_state.thread_list:
        load_thread_list()

    threads = st.session_state.thread_list
    if not threads:
        st.caption("暂无历史会话")
    else:
        for t in threads:
            tid = t["thread_id"]
            preview = t.get("preview", "") or "(空会话)"
            count = t.get("message_count", 0)
            is_active = tid == st.session_state.thread_id

            col_main, col_del = st.columns([5, 1])
            with col_main:
                label = f"{'🔵 ' if is_active else '💬 '}{preview[:25]}"
                if len(preview) > 25:
                    label += "…"
                label += f"  ({count})"
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"th_{tid}", use_container_width=True, type=btn_type):
                    st.session_state.delete_confirm = None
                    load_history(tid)

            with col_del:
                if st.button("🗑", key=f"del_{tid}", help="删除此会话"):
                    st.session_state.delete_confirm = tid

            if st.session_state.delete_confirm == tid:
                st.warning(f"确定删除「{preview[:20]}…」？")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确认", key=f"confirm_{tid}", use_container_width=True):
                        delete_thread(tid)
                        st.session_state.delete_confirm = None
                        st.rerun()
                with c2:
                    if st.button("❌ 取消", key=f"cancel_{tid}", use_container_width=True):
                        st.session_state.delete_confirm = None
                        st.rerun()

# ===================== 主界面 =====================
st.title("📊 Data Agent")
st.caption("上传数据，用自然语言分析 · 自动生成图表")

if st.session_state.file_id:
    st.info(f"📎 当前数据文件已加载（ID: {st.session_state.file_id[:8]}…）")

# --- 历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("process_steps"):
            with st.expander("🔍 查看分析过程", expanded=False):
                for ps in msg["process_steps"]:
                    st.caption(f"**{ps['step']}**")
                    st.code(ps['detail'], language=None)
        chart_bytes = decode_chart(msg.get("chart"))
        if chart_bytes:
            st.image(chart_bytes)

# --- 输入 ---
if prompt := st.chat_input("输入你的分析需求…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("分析中…"):
            payload = {
                "thread_id": st.session_state.thread_id,
                "message": prompt,
                "file_id": st.session_state.file_id
            }
            try:
                resp = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["reply"]
                    chart = data.get("chart")
                    process_steps = data.get("process_steps", [])
                    st.markdown(reply)
                    # 分析完成后展示过程（可收起）
                    if process_steps:
                        with st.expander("🔍 查看分析过程", expanded=False):
                            for ps in process_steps:
                                st.caption(f"**{ps['step']}**")
                                st.code(ps['detail'], language=None)
                    chart_bytes = decode_chart(chart)
                    if chart_bytes:
                        st.image(chart_bytes)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "chart": chart,
                        "process_steps": process_steps
                    })
                    load_thread_list()
                else:
                    st.error("请求失败，请重试")
            except requests.Timeout:
                st.error("请求超时，请重试")
            except Exception as e:
                st.error(f"连接异常: {e}")
