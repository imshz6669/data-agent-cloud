# ===== 必须在所有 import 之前：下载中文字体 =====
import os
from pathlib import Path
import requests

FONT_DIR = Path("./fonts")
FONT_PATH = FONT_DIR / "wqy-zenhei.ttc"
FONT_URL = "https://github.com/chenqinghe/fonts/raw/master/wqy-zenhei/wqy-zenhei.ttc"

if not FONT_DIR.exists():
    FONT_DIR.mkdir()
if not FONT_PATH.exists():
    print("正在下载中文字体...")
    resp = requests.get(FONT_URL, timeout=60)
    with open(FONT_PATH, "wb") as f:
        f.write(resp.content)
    print("中文字体下载完成")

os.environ["CN_FONT_PATH"] = str(FONT_PATH)
print(f"中文字体路径: {FONT_PATH}")
# ===== 字体下载完毕 =====

import sys
import uuid
import base64
import re
import streamlit as st

# Streamlit Cloud secrets → 环境变量
if "DEEPSEEK_API_KEY" in st.secrets:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]

# 确保 backend 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools import DATA_STORE, parse_uploaded_file
from backend.agent import process_chat

st.set_page_config(page_title="Data Agent — 智能数据分析", layout="wide", page_icon="📊")

# CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===================== 会话状态 =====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


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


# ===================== 侧边栏 =====================
with st.sidebar:
    st.markdown("## 📊 Data Agent")
    st.caption("智能数据分析助手 · Cloud")

    st.divider()

    if st.button("➕ 新对话", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 📁 数据上传")
    uploaded_file = st.file_uploader(
        "支持 CSV / Excel / Word",
        type=["csv", "xlsx", "xls", "docx"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        with st.spinner("解析文件中..."):
            try:
                df = parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
                file_id = str(uuid.uuid4())
                DATA_STORE[file_id] = df
                st.session_state.file_id = file_id
                st.success(f"✅ {uploaded_file.name} 已加载")
                st.caption(f"行数: {len(df)} · 列数: {len(df.columns)}")
            except Exception as e:
                st.error(f"上传失败: {e}")

    st.divider()

    if st.session_state.file_id:
        st.caption("📎 当前数据已加载")
    else:
        st.caption("⚠️ 请先上传数据文件")

# ===================== 主界面 =====================
st.title("📊 Data Agent")
st.caption("上传数据，用自然语言分析 · 自动生成图表")

if st.session_state.file_id:
    st.info("📎 数据已就绪，可直接提问分析")
else:
    st.warning("👈 请先在左侧上传 CSV / Excel / Word 文件")

# --- 历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("process_steps"):
            with st.expander("🔍 查看分析过程", expanded=False):
                for ps in msg["process_steps"]:
                    st.caption(f"**{ps['step']}**")
                    st.code(ps['detail'], language=None)
        charts = msg.get("charts", [])
        if not charts:
            chart_bytes = decode_chart(msg.get("chart"))
            if chart_bytes:
                st.image(chart_bytes)
        else:
            for chart_b64 in charts:
                chart_bytes = decode_chart(chart_b64)
                if chart_bytes:
                    st.image(chart_bytes)

# --- 输入 ---
if prompt := st.chat_input("输入你的分析需求…"):
    if not st.session_state.file_id:
        st.error("请先上传数据文件！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("分析中…"):
                try:
                    user_message = f"[当前文件ID: {st.session_state.file_id}] {prompt}"
                    result = process_chat(st.session_state.thread_id, user_message)

                    reply = result["reply"]
                    charts = result.get("charts", [])
                    process_steps = result.get("process_steps", [])

                    st.markdown(reply)
                    if process_steps:
                        with st.expander("🔍 查看分析过程", expanded=False):
                            for ps in process_steps:
                                st.caption(f"**{ps['step']}**")
                                st.code(ps['detail'], language=None)
                    for chart_b64 in charts:
                        chart_bytes = decode_chart(chart_b64)
                        if chart_bytes:
                            st.image(chart_bytes)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "charts": charts,
                        "process_steps": process_steps
                    })
                except Exception as e:
                    st.error(f"处理异常: {e}")
