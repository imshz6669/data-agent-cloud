import os
import re
import sqlite3
from typing import Dict, Any, Optional, Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# ----- 中文字体 -----
import matplotlib.font_manager as fm
import os as _os
_win_font_dir = r"C:\Windows\Fonts"
_CN_FONT_FILES = ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "simsunb.ttf", "simkai.ttf", "simfang.ttf"]
if _os.path.exists(_win_font_dir):
    for _fn in _CN_FONT_FILES:
        _fp = _os.path.join(_win_font_dir, _fn)
        if _os.path.exists(_fp):
            try:
                fm.fontManager.addfont(_fp)
            except Exception:
                pass
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_CN_FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans CJK', 'Noto Serif CJK SC',
]
_CN_FONT = next((f for f in _CN_FONT_CANDIDATES if f in _available_fonts), None)
if _CN_FONT:
    plt.rcParams['font.sans-serif'] = [_CN_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['axes.unicode_minus'] = False

from .tools import DATA_STORE, execute_nl2sql, analyze_data

# ========== 自定义 State：chart_spec 每次覆盖不累积 ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: int  # create_react_agent 要求的字段
    chart_spec: str       # 普通 str，不会被累积，每次直接覆盖

# ========== LLM ==========
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    request_timeout=120,
    max_retries=1,
)

# ========== 工具 ==========

@tool
def load_data(file_id: str) -> str:
    """加载已上传的文件数据，返回前10行预览。"""
    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：未找到该文件。"
    return df.head(10).to_string()


@tool
def query_data(file_id: str, question: str) -> str:
    """用自然语言查询数据，内部转 SQL 执行。"""
    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：文件不存在。"
    sql, result = execute_nl2sql(df, question, llm)
    return f"SQL:\n{sql}\n\n结果:\n{result.to_string(index=False)}"


@tool
def compute_data(file_id: str, code: str) -> str:
    """对数据执行 Pandas 分析代码，df 表示数据框，结果赋值给 result。"""
    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：文件不存在。"
    result_df = analyze_data(df.copy(), code)
    return result_df.to_string()


def _extract_chart_type(msg: str) -> str:
    """从用户消息提取图表类型"""
    m = msg.lower()
    if any(k in m for k in ["柱状图", "柱形图", "bar", "条形图", "对比", "排名"]):
        return "bar"
    if any(k in m for k in ["折线图", "曲线图", "line", "趋势", "走势", "变化"]):
        return "line"
    if any(k in m for k in ["饼图", "pie", "扇形", "占比", "比例", "构成"]):
        return "pie"
    if any(k in m for k in ["散点图", "scatter", "相关", "分布"]):
        return "scatter"
    if any(k in m for k in ["直方图", "histogram", "hist"]):
        return "hist"
    if any(k in m for k in ["箱线图", "箱形图", "boxplot", "box"]):
        return "box"
    return ""


@tool
def create_chart(file_id: str, chart_description: str) -> str:
    """根据描述生成图表。chart_spec 已由系统自动从用户消息提取，请使用 chart_description 传用户原文。"""
    import builtins

    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：文件不存在。"

    prompt = (
        "你是数据可视化专家。根据数据和描述写出 Matplotlib 绘图代码。\n"
        f"数据列名: {list(df.columns)}\n"
        f"数据前5行:\n{df.head().to_string()}\n"
        f"图表描述: {chart_description}\n\n"
        "要求: 只生成 Python 代码，不要解释。必须将 base64 赋值给 img_base64。\n"
        f"⚠️ 中文标签前必须设置: plt.rcParams['font.sans-serif'] = {[_CN_FONT] if _CN_FONT else ['DejaVu Sans']}\n"
        "代码模板:\n"
        "```python\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import io, base64\n"
        f"plt.rcParams['font.sans-serif'] = {[_CN_FONT] if _CN_FONT else ['DejaVu Sans']}\n"
        "plt.rcParams['axes.unicode_minus'] = False\n"
        "# 绘图代码\n"
        "buf = io.BytesIO()\n"
        "plt.savefig(buf, format='png', bbox_inches='tight')\n"
        "plt.close()\n"
        "buf.seek(0)\n"
        "img_base64 = base64.b64encode(buf.read()).decode('utf-8')\n"
        "```"
    )

    resp = llm.invoke(prompt)
    code = resp.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'io': io, 'base64': base64}
    try:
        exec(code, {"__builtins__": builtins.__dict__}, local_vars)
        img = local_vars.get('img_base64', '')
        if not img:
            return "图表生成失败：未产生图片。"
        return f"CHART_IMG:{img}"
    except Exception as e:
        return f"图表生成错误: {str(e)}"


tools = [load_data, query_data, compute_data, create_chart]

# ========== 构建 Agent ==========
DB_PATH = "agent_memory_v2.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=DELETE")
checkpointer = SqliteSaver(conn)

SYSTEM_PROMPT = (
    "你是专业的数据分析助手，可以加载数据、查询、计算、生成图表。请始终优先使用工具，最后用中文总结。\n\n"
    "图表规格(chart_spec)已自动从用户消息提取并存入State。chart_spec值为空表示用户没要求画图，"
    "值为 bar/line/pie/scatter/hist/box 时表示用户指定了对应图表类型。"
    "生成图表时请参考 chart_spec 选择合适的图表类型。"
    "每次处理用户提问时都必须独立处理，以当前用户消息中的明确指示为准。"
)

# 用 create_react_agent 构建
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    state_schema=AgentState,
    prompt=SYSTEM_PROMPT,
)

# ========== 记忆压缩 ==========
SUMMARY_THRESHOLD = 15
KEEP_RECENT = 10

def compress_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    if not state or not state.values:
        return
    messages = state.values.get("messages", [])
    if len(messages) <= SUMMARY_THRESHOLD:
        return
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    dialog_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(dialog_msgs) <= KEEP_RECENT:
        return
    recent = dialog_msgs[-KEEP_RECENT:]
    old = dialog_msgs[:-KEEP_RECENT]
    summary_prompt = "用中文总结对话历史：\n" + "\n".join(
        [f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in old]
    )
    try:
        summary_response = llm.invoke(summary_prompt)
        summary_msg = SystemMessage(content=f"[摘要] {summary_response.content}")
    except Exception:
        return
    agent.update_state(config, {"messages": system_msgs + [summary_msg] + recent})

# ========== 对话接口 ==========

def process_chat(thread_id: str, user_message: str) -> Dict[str, Any]:
    compress_history(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    # 从当前用户消息提取 chart_spec（每次覆盖，不累积）
    chart_spec = _extract_chart_type(user_message)

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)], "chart_spec": chart_spec, "remaining_steps": 20},
        config=config
    )

    process_steps = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "?")
                args = {k: str(v)[:80] for k, v in tc.get("args", {}).items()}
                process_steps.append({"step": f"🔧 {name}", "detail": str(args)})
        elif hasattr(msg, "type") and msg.type == "tool":
            content = str(msg.content)
            if content.startswith("CHART_IMG:"):
                process_steps.append({"step": "📊 图表", "detail": "已生成"})
            elif "错误" in content or "失败" in content:
                process_steps.append({"step": "⚠️ 异常", "detail": content[:200]})
            else:
                process_steps.append({"step": "✅ 结果", "detail": content[:200]})

    chart_base64 = None
    for msg in result.get("messages", []):
        if hasattr(msg, "type") and msg.type == "tool":
            m = re.search(r'CHART_IMG:([A-Za-z0-9+/=]+)', str(msg.content))
            if m:
                chart_base64 = f"data:image/png;base64,{m.group(1)}"
                # 不 break，取最后一张图（确保是当前请求生成的）

    last_msg = result["messages"][-1] if result.get("messages") else HumanMessage(content="")
    reply = last_msg.content if hasattr(last_msg, "content") else ""
    reply = re.sub(r'!\[.*?\]\(CHART_IMG[^)]*\)', '', reply)
    reply = re.sub(r'CHART_IMG:?\s*', '', reply)
    reply = re.sub(r'\n\s*\n', '\n\n', reply).strip()

    return {"reply": reply, "chart": chart_base64, "process_steps": process_steps}
