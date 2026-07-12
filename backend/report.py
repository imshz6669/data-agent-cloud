"""
模板化报告管道（Task 2）。
预计算统计 → matplotlib 模板图表 → LLM 结论。
不依赖 LLM 生成代码，稳定且快 10 倍。
"""
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# ─── 确保中文字体 ───────────────────────────────────────────
try:
    plt.rcParams.update({
        "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
#  统计计算
# ══════════════════════════════════════════════════════════════

def compute_statistics(df: pd.DataFrame, time_col: str = "") -> Dict[str, Any]:
    """预计算所有统计指标。

    Args:
        df: 数据 DataFrame
        time_col: 时间列名（空字符串表示无时间列）

    Returns:
        {"shape": ..., "numeric": {...}, "categorical": {...}, "time_agg": {...}}
    """
    stats: dict = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "rows": len(df),
    }

    # ── 数值列 ──
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe()
        stats["numeric_cols"] = num_cols
        stats["numeric"] = {
            col: {
                "count": int(desc.loc["count", col]),
                "mean": round(float(desc.loc["mean", col]), 2),
                "std": round(float(desc.loc["std", col]), 2),
                "min": round(float(desc.loc["min", col]), 2),
                "25%": round(float(desc.loc["25%", col]), 2),
                "50%": round(float(desc.loc["50%", col]), 2),
                "75%": round(float(desc.loc["75%", col]), 2),
                "max": round(float(desc.loc["max", col]), 2),
                "sum": round(float(df[col].sum()), 2),
            }
            for col in num_cols
        }

    # ── 分类列 ──
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        stats["categorical"] = {}
        for col in cat_cols[:5]:
            vc = df[col].value_counts().head(10)
            stats["categorical"][col] = {str(k): int(v) for k, v in vc.items()}

    # ── 时间聚合（按月汇总数值列）──
    if time_col and time_col in df.columns:
        try:
            dt = pd.to_datetime(df[time_col], errors="coerce")
            df_tmp = df.copy()
            df_tmp["_period"] = dt.dt.to_period("M").astype(str)
            df_valid = df_tmp.dropna(subset=["_period"])
            if not df_valid.empty and num_cols:
                agg = df_valid.groupby("_period")[num_cols].agg(["sum", "mean"]).reset_index()
                stats["time_agg"] = {
                    "periods": df_valid["_period"].tolist(),
                    "columns": num_cols,
                    "data": agg.to_dict(orient="records"),
                }
        except Exception:
            pass

    return stats


# ══════════════════════════════════════════════════════════════
#  图表生成（matplotlib 模板，非 LLM）
# ══════════════════════════════════════════════════════════════

def generate_charts(df: pd.DataFrame, stats: Dict, time_col: str = "") -> List[str]:
    """用预计算数据生成图表，返回 base64 data URL 列表。

    图表策略：
    1. 有时间列 + 数值列 → 趋势折线图
    2. 有分类列 → 柱状图
    3. 至少生成 1 张图
    """
    charts: list = []

    # ── 图表 1：趋势折线图 ──
    if time_col and stats.get("time_agg") and stats.get("numeric_cols"):
        ta = stats["time_agg"]
        # 取第一个数值列作为主指标
        main_col = stats["numeric_cols"][0]
        periods = [r["_period"] for r in ta["data"]]
        sums = [r.get((main_col, "sum"), r.get(main_col, 0)) for r in ta["data"]]

        if len(periods) >= 2:
            img = _line_chart(periods, sums, main_col, f"{main_col} 月度趋势")
            if img:
                charts.append(img)

    # ── 图表 2：分类柱状图 ──
    cat_info = stats.get("categorical", {})
    if cat_info:
        # 取第一个分类列
        col_name = list(cat_info.keys())[0]
        vc = cat_info[col_name]
        labels = list(vc.keys())[:10]
        values = list(vc.values())[:10]
        img = _bar_chart(labels, values, col_name, f"{col_name} 分布")
        if img:
            charts.append(img)

    # ── 兜底：如果没有生成任何图表，生成一张汇总柱状图 ──
    if not charts and stats.get("numeric_cols"):
        nums = stats["numeric"]
        main_col = stats["numeric_cols"][0]
        n = nums[main_col]
        img = _summary_bar(main_col, n)
        if img:
            charts.append(img)

    return charts


def _fig_to_b64(fig) -> str:
    """将 matplotlib Figure 转为 base64 data URL。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


def _line_chart(x: list, y: list, ylabel: str, title: str) -> Optional[str]:
    """趋势折线图模板。"""
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(len(x)), y, "o-", color="#2563EB", linewidth=2, markersize=6)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 数值标注
        for i, v in enumerate(y):
            ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#1E40AF")

        try:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
        except Exception:
            pass

        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None


def _bar_chart(labels: list, values: list, xlabel: str, title: str) -> Optional[str]:
    """柱状图模板。"""
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(values)))
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.15 if values else 1)

        # bar_label
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:,}", ha="center", va="bottom", fontsize=9, color="#1E40AF")

        try:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
        except Exception:
            pass

        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None


def _summary_bar(col_name: str, stats: dict) -> Optional[str]:
    """汇总统计柱状图（兜底）。"""
    try:
        metrics = ["mean", "min", "50%", "max", "sum"]
        labels = ["均值", "最小值", "中位数", "最大值", "总和"]
        values = [stats.get(m, 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#60A5FA", "#34D399", "#FBBF24", "#F87171", "#A78BFA"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=10)

        ax.set_title(f"{col_name} 统计概览", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        try:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
        except Exception:
            pass

        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  结论生成（LLM 只看统计结果写结论）
# ══════════════════════════════════════════════════════════════

CONCLUSION_PROMPT = """你是数据分析师。根据以下数据统计，用中文写一段简短的分析结论（80~120字）。

数据概况：
- 行数: {rows}
- 列数: {cols}
- 列名: {columns}{numeric_summary}{categorical_summary}{time_summary}

要求：只输出结论文字，不要标题、编号或额外解释。"""


def build_conclusion_prompt(stats: dict) -> str:
    """构建 LLM 结论 prompt。"""
    numeric_summary = ""
    if stats.get("numeric"):
        lines = []
        for col, n in stats["numeric"].items():
            lines.append(f"  {col}: 均值={n['mean']}, 总和={n['sum']}, 最大={n['max']}")
        numeric_summary = "\n数值指标:\n" + "\n".join(lines)

    categorical_summary = ""
    if stats.get("categorical"):
        lines = []
        for col, vc in stats["categorical"].items():
            top3 = list(vc.items())[:3]
            lines.append(f"  {col}: {', '.join(f'{k}({v})' for k, v in top3)}")
        categorical_summary = "\n分类分布:\n" + "\n".join(lines)

    time_summary = ""
    if stats.get("time_agg"):
        time_summary = f"\n时间范围: {len(stats['time_agg']['periods'])} 个月"

    return CONCLUSION_PROMPT.format(
        rows=stats["rows"],
        cols=len(stats["columns"]),
        columns=", ".join(stats["columns"]),
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        time_summary=time_summary,
    )


def generate_conclusion(stats: dict, chart_count: int, llm) -> str:
    """让 LLM 根据统计结果生成结论。

    Args:
        stats: compute_statistics 的输出
        chart_count: 生成的图表数量
        llm: LangChain ChatOpenAI 实例
    """
    prompt = build_conclusion_prompt(stats)
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return _fallback_conclusion(stats)


def _fallback_conclusion(stats: dict) -> str:
    """LLM 不可用时的兜底结论。"""
    parts = [f"共 {stats['rows']} 条记录"]
    if stats.get("numeric"):
        for col, n in stats["numeric"].items():
            parts.append(f"{col} 均值 {n['mean']}，总和 {n['sum']}")
    return "。".join(parts) + "。"
