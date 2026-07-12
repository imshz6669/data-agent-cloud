"""
模板化报告管道（Task 2 重写）。
预计算统计 → matplotlib 模板图表 → LLM 结论。
每张图表生成前强制设置中文字体，解决 Streamlit Cloud 乱码。
"""
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


# ─── 中文字体（每张图表前强制执行）──────────────────────────

def _set_font():
    """每张图表生成前强制设置中文字体。"""
    plt.rcParams.update({
        "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
    })


# ══════════════════════════════════════════════════════════════
#  列名探测
# ══════════════════════════════════════════════════════════════

_SALES_KEYS = ["销售", "金额", "收入", "sales", "amount", "revenue", "营收", "总价"]
_QTY_KEYS = ["数量", "订单", "销量", "quantity", "orders", "qty", "件数", "笔数"]
_PRICE_KEYS = ["单价", "均价", "price", "客单价"]
_CAT_KEYS = ["品类", "类别", "分类", "category", "类型", "产品", "product",
             "区域", "地区", "region", "渠道", "channel", "部门", "品牌"]


def _match(col: str, keys: list) -> bool:
    return any(k in str(col).lower() for k in keys)


def _detect_columns(df: pd.DataFrame) -> dict:
    """探测列类型。"""
    info = {
        "sales_cols": [],
        "qty_cols": [],
        "price_cols": [],
        "cat_cols": [],
        "numeric_cols": [],
        "time_cols": [],
    }
    for col in df.columns:
        cl = str(col).lower()
        if _match(cl, _SALES_KEYS):
            info["sales_cols"].append(col)
        elif _match(cl, _QTY_KEYS):
            info["qty_cols"].append(col)
        elif _match(cl, _PRICE_KEYS):
            info["price_cols"].append(col)
        if _match(cl, _CAT_KEYS):
            info["cat_cols"].append(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            info["numeric_cols"].append(col)
    # 去重
    for key in ["sales_cols", "qty_cols", "price_cols"]:
        info[key] = list(dict.fromkeys(info[key]))
    return info


# ══════════════════════════════════════════════════════════════
#  报告生成主函数
# ══════════════════════════════════════════════════════════════

def generate_full_report(
    df: pd.DataFrame,
    file_name: str = "",
    time_col: str = "",
    llm=None,
) -> dict:
    """生成完整分析报告。

    Returns:
        {"reply": str, "charts": [data_url, ...], "process_steps": [...]}
    """
    meta = _detect_columns(df)
    charts: list = []
    steps: list = []

    # ── 1. 数据概览 ──
    rows = len(df)
    period = _detect_period(df, time_col) if time_col else ""
    overview = f"## 📊 数据概览\n\n"
    if period:
        overview += f"- **报告周期**：{period}\n"
    if file_name:
        overview += f"- **数据来源**：{file_name}\n"
    overview += f"- **数据行数**：{rows} 条\n"
    overview += f"- **数据列数**：{len(df.columns)} 列\n"
    steps.append({"step": "📈 数据概览", "detail": f"{rows}行 × {len(df.columns)}列"})

    # ── 2. 核心指标卡 ──
    kpi_lines = []
    kpi_text = ""

    # 总销售额
    sales_col = meta["sales_cols"][0] if meta["sales_cols"] else (
        meta["numeric_cols"][0] if meta["numeric_cols"] else None
    )
    total_sales = None
    if sales_col:
        total_sales = round(float(df[sales_col].sum()), 2)
        kpi_lines.append(f"| 💰 总销售额 | **{total_sales:,.2f}** |")
        kpi_text += f"总销售额 {total_sales:,.2f}"

    # 总订单量
    qty_col = meta["qty_cols"][0] if meta["qty_cols"] else None
    total_qty = None
    if qty_col:
        total_qty = int(df[qty_col].sum())
        kpi_lines.append(f"| 📦 总订单量 | **{total_qty:,}** |")
        kpi_text += f"，总订单量 {total_qty:,}"

    # 平均客单价
    avg_price = None
    if total_sales and total_qty and total_qty > 0:
        avg_price = round(total_sales / total_qty, 2)
        kpi_lines.append(f"| 🛒 平均客单价 | **{avg_price:,.2f}** |")
        kpi_text += f"，平均客单价 {avg_price:,.2f}"

    # 环比/同比（有时间列且 >= 2 个周期时计算）
    mom_text = ""
    if time_col and total_sales:
        mom = _compute_mom(df, time_col, sales_col)
        if mom is not None:
            arrow = "↑" if mom > 0 else "↓" if mom < 0 else "→"
            kpi_lines.append(f"| 📈 环比变化 | **{arrow} {abs(mom):.1f}%** |")
            mom_text = f"，环比{mom:+.1f}%"

    if kpi_lines:
        kpi_section = "## 📊 核心指标\n\n| 指标 | 数值 |\n|------|------|\n" + "\n".join(kpi_lines) + "\n"
    else:
        kpi_section = ""

    if kpi_text:
        overview += f"\n> **关键结论**：{kpi_text}{mom_text}。\n"
    steps.append({"step": "📊 核心指标", "detail": kpi_text[:100] if kpi_text else "已计算"})

    # ── 3. 趋势可视化 ──
    if time_col and sales_col and meta["numeric_cols"]:
        periods = sorted(set(pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").dropna().astype(str)))
        if len(periods) >= 2:
            # 按月汇总
            chart_df = df.copy()
            chart_df["_p"] = pd.to_datetime(chart_df[time_col], errors="coerce").dt.to_period("M").astype(str)
            monthly = chart_df.groupby("_p")[sales_col].sum().reset_index()
            monthly = monthly[monthly["_p"].isin(periods[-12:])]  # 最多 12 个月

            img = _trend_chart(monthly["_p"].tolist(), monthly[sales_col].tolist(), sales_col)
            if img:
                charts.append(img)
                steps.append({"step": "📈 趋势图", "detail": f"{sales_col} 月度趋势（{len(periods)}个月）"})

    # ── 4. 分类对比图 ──
    cat_col = meta["cat_cols"][0] if meta["cat_cols"] else None
    if cat_col and sales_col:
        top = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False).head(10)
        img = _bar_chart(top.index.tolist(), top.values.tolist(), cat_col, sales_col)
        if img:
            charts.append(img)
            steps.append({"step": "📊 分类对比", "detail": f"{cat_col} × {sales_col} Top10"})

    # ── 5. AI 关键发现 ──
    findings = ""
    if llm:
        findings = _generate_findings(df, meta, kpi_text, mom_text, time_col, llm)
        if findings:
            steps.append({"step": "📝 AI 洞察", "detail": findings[:100]})

    # ── 拼接最终回复 ──
    reply = overview + "\n" + kpi_section
    if findings:
        reply += f"\n## 🔍 关键发现\n\n{findings}\n"

    return {
        "reply": reply.strip(),
        "charts": charts,
        "process_steps": steps,
    }


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def _detect_period(df: pd.DataFrame, time_col: str) -> str:
    """检测报告周期。"""
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if len(dt) == 0:
            return ""
        p = dt.dt.to_period("M")
        vals = sorted(set(p.dropna().astype(str)))
        if len(vals) == 1:
            return vals[0]
        return f"{vals[0]} 至 {vals[-1]}（{len(vals)}个月）"
    except Exception:
        return ""


def _compute_mom(df: pd.DataFrame, time_col: str, value_col: str) -> Optional[float]:
    """计算环比变化（最近两期对比）。"""
    try:
        chart_df = df.copy()
        chart_df["_p"] = pd.to_datetime(chart_df[time_col], errors="coerce").dt.to_period("M").astype(str)
        monthly = chart_df.groupby("_p")[value_col].sum().sort_index()
        if len(monthly) < 2:
            return None
        curr = monthly.iloc[-1]
        prev = monthly.iloc[-2]
        if prev == 0:
            return None
        return round((curr - prev) / prev * 100, 1)
    except Exception:
        return None


def _generate_findings(df, meta, kpi_text, mom_text, time_col, llm) -> str:
    """AI 生成关键发现。"""
    prompt = f"""你是数据分析师。根据以下信息写一段简短的分析发现（100-150字）。

数据规模：{len(df)} 行 × {len(df.columns)} 列
列名：{', '.join(df.columns[:15])}
核心指标：{kpi_text}{mom_text}

请直接输出发现，不要标题，内容应包括：
- 表现最好的维度（如果有分类列）
- 值得关注的异常或趋势
- 一个简短建议"""

    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
#  图表模板（每张图内强制设字体）
# ══════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def _trend_chart(x: list, y: list, ylabel: str) -> Optional[str]:
    """月度趋势折线图。"""
    try:
        _set_font()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(x)), y, "o-", color="#2563EB", linewidth=2.5, markersize=7)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} 月度趋势", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 标注最高点和最低点
        max_i = np.argmax(y)
        min_i = np.argmin(y)
        ax.annotate(f"最高 {y[max_i]:,.0f}", (max_i, y[max_i]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color="#DC2626", fontweight="bold")
        ax.annotate(f"最低 {y[min_i]:,.0f}", (min_i, y[min_i]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, color="#059669", fontweight="bold")

        for i, v in enumerate(y):
            ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#1E40AF")

        try:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
        except Exception:
            pass

        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None


def _bar_chart(labels: list, values: list, xlabel: str, ylabel: str) -> Optional[str]:
    """分类柱状图。"""
    try:
        _set_font()
        n = min(len(labels), 10)
        labels = labels[:n]
        values = values[:n]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Blues(np.linspace(0.35, 0.9, n))
        bars = ax.bar(range(n), values, color=colors, edgecolor="white", width=0.65)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{xlabel} × {ylabel} Top{n}", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        if values:
            ax.set_ylim(0, max(values) * 1.18)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color="#1E40AF")

        try:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
        except Exception:
            pass

        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None
