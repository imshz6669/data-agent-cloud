# ===== 必须在所有 import 之前：字体 & 后端初始化 =====
import os as _os
import glob as _glob

# Streamlit Cloud 上 matplotlib 缓存目录可能损坏导致 segfault
_os.environ["MPLCONFIGDIR"] = "/tmp/mpl-cache"
for _f in _glob.glob("/tmp/mpl-cache/fontlist-*.json"):
    try:
        _os.remove(_f)
    except Exception:
        pass
for _f in _glob.glob(_os.path.expanduser("~/.cache/matplotlib/fontlist-*.json")):
    try:
        _os.remove(_f)
    except Exception:
        pass

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib

matplotlib.use("Agg")  # Streamlit 无图形界面，必须指定后端

import matplotlib.pyplot as plt

# 全局设置中文字体（packages.txt 保证 fonts-wqy-zenhei 已安装）
try:
    plt.rcParams.update(
        {
            "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.family": "sans-serif",
        }
    )
except Exception:
    pass
# ===== 字体初始化完毕 =====

import sys
import os
import uuid
import base64
import re
from datetime import datetime

import streamlit as st

# Streamlit Cloud secrets → 环境变量
if "DEEPSEEK_API_KEY" in st.secrets:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]

# 确保 backend 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools import DATA_STORE, parse_uploaded_file
from backend.agent import process_chat, llm
from backend.dimension import extract_dimensions, apply_filters
from backend.report import (
    compute_statistics,
    generate_charts,
    generate_conclusion,
)
from backend.export import create_zip_export, create_pdf_export

st.set_page_config(
    page_title="Data Agent — 智能数据分析", layout="wide", page_icon="📊"
)

# CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  会话状态初始化
# ══════════════════════════════════════════════════════════════
    st.session_state.thread_id = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 维度过滤
if "full_df" not in st.session_state:
    st.session_state.full_df = None
if "dimensions" not in st.session_state:
    st.session_state.dimensions = {"time": [], "category": []}
if "time_filters" not in st.session_state:
    st.session_state.time_filters = {}
if "category_filters" not in st.session_state:
    st.session_state.category_filters = {}

# 报告管道
if "_report_running" not in st.session_state:
    st.session_state._report_running = False
if "_report_stage" not in st.session_state:
    st.session_state._report_stage = 0
if "_report_data" not in st.session_state:
    st.session_state._report_data = {}

# 自动导出
if "auto_export" not in st.session_state:
    st.session_state.auto_export = False
if "_export_zip" not in st.session_state:
    st.session_state._export_zip = None
if "_export_pdf" not in st.session_state:
    st.session_state._export_pdf = None


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

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


def _apply_all_filters():
    """将当前维度选择应用到 DATA_STORE。"""
    fid = st.session_state.file_id
    if not fid or st.session_state.full_df is None:
        return

    has_time = any(v for v in st.session_state.time_filters.values())
    has_cat = any(v for v in st.session_state.category_filters.values())

    if has_time or has_cat:
        filtered = apply_filters(
            st.session_state.full_df.copy(),
            st.session_state.time_filters,
            st.session_state.category_filters,
        )
        DATA_STORE[fid] = filtered
    else:
        DATA_STORE[fid] = st.session_state.full_df.copy()


def _run_export():
    """执行导出，更新 _export_zip / _export_pdf。"""
    msgs = st.session_state.messages
    fname = st.session_state.get("file_name", "")
    if msgs:
        st.session_state._export_zip = create_zip_export(msgs, fname)
        st.session_state._export_pdf = create_pdf_export(msgs, fname)


# ══════════════════════════════════════════════════════════════
#  报告管道阶段
# ══════════════════════════════════════════════════════════════

REPORT_STAGES = {
    1: ("📈 计算统计指标", "stats"),
    2: ("📊 生成图表", "charts"),
    3: ("📝 AI 撰写结论", "conclusion"),
    4: ("✅ 完成", "done"),
}


# ══════════════════════════════════════════════════════════════
#  侧边栏
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Data Agent")
    st.caption("智能数据分析助手 · Cloud")

    st.divider()

    # ── 新对话 ──
    if st.button("➕ 新对话", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.time_filters = {}
        st.session_state.category_filters = {}
        st.session_state._report_running = False
        st.session_state._report_stage = 0
        st.session_state._report_data = {}
        st.session_state._export_zip = None
        st.session_state._export_pdf = None
        if st.session_state.full_df is not None and st.session_state.file_id:
            DATA_STORE[st.session_state.file_id] = st.session_state.full_df.copy()
        st.rerun()

    st.divider()

    # ── 文件上传 ──
    st.markdown("### 📁 数据上传")
    uploaded_file = st.file_uploader(
        "支持 CSV / Excel / Word",
        type=["csv", "xlsx", "xls", "docx"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        with st.spinner("解析文件中..."):
            try:
                df = parse_uploaded_file(
                    uploaded_file.getvalue(), uploaded_file.name
                )
                file_id = str(uuid.uuid4())
                DATA_STORE[file_id] = df
                st.session_state.file_id = file_id
                st.session_state.file_name = uploaded_file.name
                st.session_state.full_df = df.copy()
                st.session_state.messages = []
                st.session_state.time_filters = {}
                st.session_state.category_filters = {}
                st.session_state._report_running = False
                st.session_state._report_stage = 0

                # 提取维度（带缓存）
                dims = extract_dimensions(df)
                st.session_state.dimensions = dims

                st.success(f"✅ {uploaded_file.name} 已加载")
                n_time = len(dims.get("time", []))
                n_cat = len(dims.get("category", []))
                st.caption(
                    f"行数: {len(df)} · 列数: {len(df.columns)}"
                    f"{' · 时间列: ' + str(n_time) + '个' if n_time else ''}"
                    f"{' · 分类列: ' + str(n_cat) + '个' if n_cat else ''}"
                )
            except Exception as e:
                st.error(f"上传失败: {e}")

    # ── 维度过滤 ──
    dims = st.session_state.get("dimensions", {})
    if st.session_state.file_id and (dims.get("time") or dims.get("category")):
        st.divider()
        st.markdown("### 🔍 维度过滤")

        changed = False

        # 时间维度
        for tc in dims.get("time", []):
            col = tc["column"]
            periods = tc["periods"]
            selected = st.multiselect(
                f"⏱ {col}（留空=全部）",
                options=periods,
                default=st.session_state.time_filters.get(col, []),
                key=f"time_filter_{col}",
            )
            if selected != st.session_state.time_filters.get(col, []):
                st.session_state.time_filters[col] = selected
                changed = True

        # 分类维度
        for cc in dims.get("category", []):
            col = cc["column"]
            values = cc["values"]
            selected = st.multiselect(
                f"🏷 {col}（留空=全部）",
                options=values,
                default=st.session_state.category_filters.get(col, []),
                key=f"cat_filter_{col}",
            )
            if selected != st.session_state.category_filters.get(col, []):
                st.session_state.category_filters[col] = selected
                changed = True

        if changed:
            _apply_all_filters()
            st.rerun()

    # ── 文件状态 ──
    st.divider()
    if st.session_state.file_id:
        st.caption("📎 当前数据已加载")
    else:
        st.caption("⚠️ 请先上传数据文件")

    # ── 自动导出开关 ──
    st.divider()
    auto = st.toggle("🔁 自动导出", value=st.session_state.auto_export)
    if auto != st.session_state.auto_export:
        st.session_state.auto_export = auto
        if auto and st.session_state.messages:
            _run_export()
        st.rerun()

    # ── 一键生成报告 ──
    st.divider()
    if st.button(
        "📊 一键生成报告",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.file_id,
    ):
        st.session_state._report_running = True
        st.session_state._report_stage = 1
        st.session_state._report_data = {}
        st.rerun()

    # ── 导出下载 ──
    if st.session_state.messages:
        st.divider()
        st.markdown("### 📥 导出报告")

        # 确保有导出文件
        if st.session_state._export_zip is None:
            _run_export()

        if st.session_state._export_zip:
            st.download_button(
                label="📦 下载 ZIP（Markdown + 图片）",
                data=st.session_state._export_zip,
                file_name=f"data_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        if st.session_state._export_pdf:
            st.download_button(
                label="📄 下载 PDF",
                data=st.session_state._export_pdf,
                file_name=f"data_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════════

st.title("📊 Data Agent")
st.caption("上传数据，用自然语言分析 · 自动生成图表")

if st.session_state.file_id:
    st.info("📎 数据已就绪，可直接提问分析")
else:
    st.warning("👈 请先在左侧上传 CSV / Excel / Word 文件")

# ── 报告管道进度 ──
if st.session_state._report_running:
    stage = st.session_state._report_stage
    total = len(REPORT_STAGES)
    label, _key = REPORT_STAGES.get(stage, (f"阶段 {stage}", ""))

    status = st.status(f"📊 正在生成报告... ({stage}/{total})", expanded=True)

    # ── 阶段 1: 统计 ──
    if stage == 1:
        status.write("📈 计算统计指标...")
        fid = st.session_state.file_id
        df = DATA_STORE.get(fid, st.session_state.full_df)
        if df is not None:
            time_col = (
                st.session_state.dimensions["time"][0]["column"]
                if st.session_state.dimensions.get("time")
                else ""
            )
            st.session_state._report_data["stats"] = compute_statistics(
                df, time_col
            )
            st.session_state._report_data["time_col"] = time_col
        st.session_state._report_stage = 2
        st.rerun()

    # ── 阶段 2: 图表 ──
    elif stage == 2:
        status.write("📊 生成图表...")
        fid = st.session_state.file_id
        df = DATA_STORE.get(fid, st.session_state.full_df)
        stats = st.session_state._report_data.get("stats", {})
        time_col = st.session_state._report_data.get("time_col", "")
        if df is not None:
            charts = generate_charts(df, stats, time_col)
            st.session_state._report_data["charts"] = charts
        st.session_state._report_stage = 3
        st.rerun()

    # ── 阶段 3: 结论 ──
    elif stage == 3:
        status.write("📝 AI 撰写结论...")
        stats = st.session_state._report_data.get("stats", {})
        charts = st.session_state._report_data.get("charts", [])
        try:
            conclusion = generate_conclusion(stats, len(charts), llm)
        except Exception:
            conclusion = f"共 {stats.get('rows', '?')} 条记录，分析完成。"
        st.session_state._report_data["conclusion"] = conclusion
        st.session_state._report_stage = 4
        st.rerun()

    # ── 阶段 4: 完成 ──
    elif stage == 4:
        status.update(label="✅ 报告生成完成！", state="complete")
        stats = st.session_state._report_data.get("stats", {})
        charts = st.session_state._report_data.get("charts", [])
        conclusion = st.session_state._report_data.get("conclusion", "")

        # 构建报告文本
        lines = ["## 📊 数据分析报告\n"]
        lines.append(f"**数据规模**：{stats.get('rows', '?')} 行 × {len(stats.get('columns', []))} 列\n")

        if stats.get("numeric"):
            lines.append("### 📈 关键指标")
            for col, n in list(stats["numeric"].items())[:5]:
                lines.append(f"- **{col}**：均值 {n['mean']}，总和 {n['sum']}，最大 {n['max']}")

        if conclusion:
            lines.append(f"\n### 📝 分析结论\n{conclusion}")

        reply = "\n".join(lines)

        st.session_state.messages.append(
            {"role": "user", "content": "📊 一键生成报告（自动）"}
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "charts": charts,
                "process_steps": [
                    {"step": "📈 统计", "detail": f"{stats.get('rows', '?')}行 × {len(stats.get('columns', []))}列"},
                    {"step": "📊 图表", "detail": f"生成 {len(charts)} 张图表"},
                    {"step": "📝 结论", "detail": conclusion[:100] + "..." if len(conclusion) > 100 else conclusion},
                ],
            }
        )

        # 自动导出
        if st.session_state.auto_export:
            _run_export()

        st.session_state._report_running = False
        st.session_state._report_stage = 0
        st.rerun()


# ── 历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("process_steps"):
            with st.expander("🔍 查看分析过程", expanded=False):
                for ps in msg["process_steps"]:
                    st.caption(f"**{ps['step']}**")
                    st.code(ps["detail"], language=None)
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


# ── 输入 ──
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
                    user_message = (
                        f"[当前文件ID: {st.session_state.file_id}] {prompt}"
                    )
                    result = process_chat(
                        st.session_state.thread_id, user_message
                    )

                    reply = result["reply"]
                    charts = result.get("charts", [])
                    process_steps = result.get("process_steps", [])

                    st.markdown(reply)
                    if process_steps:
                        with st.expander("🔍 查看分析过程", expanded=False):
                            for ps in process_steps:
                                st.caption(f"**{ps['step']}**")
                                st.code(ps["detail"], language=None)
                    for chart_b64 in charts:
                        chart_bytes = decode_chart(chart_b64)
                        if chart_bytes:
                            st.image(chart_bytes)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "charts": charts,
                            "process_steps": process_steps,
                        }
                    )

                    # 自动导出
                    if st.session_state.auto_export:
                        _run_export()
                        st.rerun()

                except Exception as e:
                    st.error(f"处理异常: {e}")
