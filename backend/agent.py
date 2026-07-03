import os
import re
from typing import Dict, Any, Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
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

# 中文字体由 frontend/app.py 在导入前全局配置，此处无需重复设置

from .tools import DATA_STORE, execute_nl2sql, analyze_data

# ========== 自定义 State ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: int
    chart_spec: str

# ========== LLM ==========
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
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
    """对数据执行 Pandas 分析代码，df 表示数据框，结果赋值给 result。禁止绘图（不要用 plt/matplotlib），只做数据计算。"""
    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：文件不存在。"
    result_df = analyze_data(df.copy(), code)
    return result_df.to_string()


def _extract_chart_type(msg: str) -> str:
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
    """根据描述生成图表。"""
    import builtins

    df = DATA_STORE.get(file_id)
    if df is None:
        return "错误：文件不存在。"

    _stats_lines = [f"数据行数: {len(df)}, 列数: {len(df.columns)}"]
    for _col in df.columns:
        _dtype = df[_col].dtype
        if _dtype in ('int64', 'float64'):
            _stats_lines.append(f"  {_col}({_dtype}): min={df[_col].min()}, max={df[_col].max()}, mean={df[_col].mean():.2f}")
        else:
            _uvals = df[_col].dropna().unique()
            _stats_lines.append(f"  {_col}({_dtype}): 唯一值={len(_uvals)}个 → {list(_uvals[:8])}{'…' if len(_uvals) > 8 else ''}")

    prompt = (
        "你是数据可视化专家。根据数据和描述写出 Matplotlib 绘图代码。\n"
        f"数据列名: {list(df.columns)}\n"
        f"数据前5行:\n{df.head().to_string()}\n"
        f"数据统计:\n" + "\n".join(_stats_lines) + "\n\n"
        f"图表描述: {chart_description}\n\n"
        "🚫 严禁 plt.savefig('xxx.png') 保存到文件！必须严格按代码模板用 io.BytesIO + base64 编码。\n"
        "⚠️ 数据约束（必须遵守）：\n"
        "  · 严格使用提供的 df 变量操作，禁止自己编造数据或硬编码数值\n"
        "  · 分类分组必须用 df 中实际存在的列，不得假设不存在的列名\n"
        "  · 聚合操作（groupby/sum/mean等）必须基于 df 的列，聚合结果与 describe 统计一致\n"
        "  · 图表中的数值必须与 df 原始数据或聚合结果完全一致，不得凭空产生\n"
        "规范要求（必须遵守，只生成 Python 代码，不要解释，base64 赋值给 img_base64）：\n"
        "  · 科学计数法: 禁止！在数值轴上用 ax.yaxis.set_major_formatter(ScalarFormatter()) 或 try: plt.ticklabel_format(style='plain', axis='y') except: pass\n"
        "  · 数值标注: 柱状图用 plt.bar_label()，折线图在数据点标注，饼图设 autopct\n"
        "  · 标题: 必须含「分析维度+指标」如\"2017-2019年各渠道销售额柱状图\"，居中，字号>轴标签\n"
        "  · X/Y轴: 标注名称+单位（如\"销售额（元）\"），分类变量显示全部分组；Y轴若为金额/数量须从0起始\n"
        "  · 图例: 多系列必须添加图例，置于右上角不遮挡数据；单系列可省略\n"
        "  · 画布: figsize=(9,6)，高区分度配色，浅灰虚线网格\n"
        "  · 柱状图: Y轴从0起，柱顶 bar_label 标注，多分组并列+图例\n"
        "  · 折线图: X轴时间序列，不同线型+颜色+拐点标记('-o')，仅用于趋势非离散对比\n"
        "  · 箱体图: 每分类独立完整箱体(含Q1/中位/须)，异常值显示，中位线加粗(lw=2)。⚠️ 参数名是 label 不是 labels，用 tick_labels 设X轴标签。\n"
        "  · 饼图: ≤6类，标注百分比\n"
        "  · 直方图: 均匀区间；散点图: 标注自变量/因变量含义\n"
        "代码模板（唯一正确输出方式，必须完全按此模式）：\n"
        "```python\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import io, base64\n"
        "plt.rcParams['axes.unicode_minus'] = False\n"
        "from matplotlib.ticker import ScalarFormatter\n"
        "try:\n"
        "    plt.gca().yaxis.set_major_formatter(ScalarFormatter())\n"
        "    plt.ticklabel_format(style='plain', axis='y')\n"
        "except Exception:\n"
        "    pass\n"
        "plt.figure(figsize=(9,6))\n"
        "plt.grid(True, linestyle='--', alpha=0.3)\n"
        "# 绘图代码\n"
        "# 箱体图: plt.boxplot(..., tick_labels=分类名) NOT labels=！参数是 tick_labels 不是 labels\n"
        "# 柱体必须 bar_label，折线必须标注数据点，饼图必须 autopct\n"
        "buf = io.BytesIO()        # ← 必须用 BytesIO，禁止 plt.savefig('文件.png')\n"
        "plt.savefig(buf, format='png', bbox_inches='tight')\n"
        "plt.close()\n"
        "buf.seek(0)\n"
        "img_base64 = base64.b64encode(buf.read()).decode('utf-8')  # ← 必须赋值 img_base64\n"
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
        # 注入字体设置代码到 LLM 生成的代码头部（确保 exec 中也不会丢字体）
        _font_path = os.environ.get("CN_FONT_PATH", "")
        if _font_path:
            _font_inject = (
                "import matplotlib.pyplot as plt\n"
                "import matplotlib.font_manager as _fm\n"
                "import warnings\n"
                "warnings.filterwarnings('ignore')\n"
                f"try:\n"
                f"    _fm.fontManager.addfont(r'{_font_path}')\n"
                f"except Exception:\n"
                f"    pass\n"
                "plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']\n"
                "plt.rcParams['axes.unicode_minus'] = False\n"
            )
            code = _font_inject + code

        _savefig_calls = re.findall(r'plt\.savefig\(([^)]+(?:\([^)]*\)[^)]*)*)\)', code)
        for _call in _savefig_calls:
            _first_arg = _call.split(',')[0].strip().strip("'").strip('"')
            if _first_arg != 'buf':
                return "错误：plt.savefig 写到了文件而非 buf！必须用 plt.savefig(buf, format='png', bbox_inches='tight')，严禁保存为磁盘文件。请修正代码。"

        import glob as _glob
        _png_before = set(_glob.glob("*.png"))

        exec(code, {"__builtins__": builtins.__dict__}, local_vars)

        for _fp in set(_glob.glob("*.png")) - _png_before:
            try:
                os.remove(_fp)
            except Exception:
                pass

        img = local_vars.get('img_base64', '')
        if not img:
            return "图表生成失败：未产生图片。"
        return f"CHART_IMG:{img}"
    except Exception as e:
        return f"图表生成错误: {str(e)}"


tools = [load_data, query_data, compute_data, create_chart]

# ========== 构建 Agent（MemorySaver，重启丢失） ==========
checkpointer = MemorySaver()

SYSTEM_PROMPT = (
    "你是专业的数据分析助手，可以加载数据、查询、计算、生成图表。请始终优先使用工具，最后用中文总结。\n\n"
    "图表规格(chart_spec)已自动从用户消息提取并存入State。chart_spec值为空表示用户没要求画图，"
    "值为 bar/line/pie/scatter/hist/box 时表示用户指定了对应图表类型。"
    "生成图表时请参考 chart_spec 选择合适的图表类型。\n"
    "⚠️ 回复中禁止提及文件路径（如'保存为 xxx.png'、'文件已保存至'等），图表已通过工具自动返回前端展示。\n"
    "每次处理用户提问时都必须独立处理，以当前用户消息中的明确指示为准。"
)

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

    chart_spec = _extract_chart_type(user_message)

    state_before = agent.get_state(config)
    msg_count_before = len(state_before.values.get("messages", [])) if state_before and state_before.values else 0

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)], "chart_spec": chart_spec, "remaining_steps": 20},
        config=config
    )

    new_messages = result.get("messages", [])[msg_count_before:]

    process_steps = []
    for msg in new_messages:
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

    charts = []
    for msg in new_messages:
        if hasattr(msg, "type") and msg.type == "tool":
            for m in re.finditer(r'CHART_IMG:([A-Za-z0-9+/=]+)', str(msg.content)):
                charts.append(f"data:image/png;base64,{m.group(1)}")

    last_msg = result["messages"][-1] if result.get("messages") else HumanMessage(content="")
    reply = last_msg.content if hasattr(last_msg, "content") else ""
    reply = re.sub(r'!\[.*?\]\(CHART_IMG[^)]*\)', '', reply)
    reply = re.sub(r'CHART_IMG:?\s*', '', reply)
    reply = re.sub(r'\n\s*\n', '\n\n', reply).strip()

    return {"reply": reply, "charts": charts, "process_steps": process_steps}
