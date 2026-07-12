"""
模板化报告管道 — 增强版
预计算统计 → matplotlib 模板图表 → LLM 结论。
每张图表生成前强制设置中文字体。
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
    plt.rcParams.update({
        "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
    })


# ══════════════════════════════════════════════════════════════
#  列名探测
# ══════════════════════════════════════════════════════════════

_SALES_KEYS = ["销售", "金额", "收入", "sales", "amount", "revenue", "营收", "总价", "成交"]
_QTY_KEYS = ["数量", "订单", "销量", "quantity", "orders", "qty", "件数", "笔数", "单数"]
_PRICE_KEYS = ["单价", "均价", "price", "客单价"]
_CAT_KEYS = ["品类", "类别", "分类", "category", "类型", "产品", "product",
             "区域", "地区", "region", "渠道", "channel", "部门", "品牌",
             "城市", "city", "门店", "store", "仓库"]


def _match(col: str, keys: list) -> bool:
    return any(k in str(col).lower() for k in keys)


def _detect_columns(df: pd.DataFrame) -> dict:
    info = {
        "sales_cols": [], "qty_cols": [], "price_cols": [],
        "cat_cols": [], "numeric_cols": [], "time_cols": [],
    }
    for col in df.columns:
        cl = str(col).lower()
        # 排除日期列（pd.to_numeric 能转日期→整数，但不应作为数值指标列）
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            info["time_cols"].append(col)
            continue

        # 用 pd.to_numeric 确认列确实可转数值，避免 object 列混入
        _sample = df[col].dropna().iloc[:10] if df[col].dropna().shape[0] > 0 else pd.Series([], dtype=object)
        _is_numeric = True
        try:
            pd.to_numeric(_sample, errors="raise")
        except (ValueError, TypeError):
            _is_numeric = False

        if _match(cl, _SALES_KEYS) and _is_numeric:
            info["sales_cols"].append(col)
        if _match(cl, _QTY_KEYS) and _is_numeric:
            info["qty_cols"].append(col)
        if _match(cl, _PRICE_KEYS) and _is_numeric:
            info["price_cols"].append(col)
        if _match(cl, _CAT_KEYS):
            info["cat_cols"].append(col)
        if _is_numeric:
            info["numeric_cols"].append(col)
    for key in ["sales_cols", "qty_cols", "price_cols"]:
        info[key] = list(dict.fromkeys(info[key]))
    return info


# ══════════════════════════════════════════════════════════════
#  LLM 列角色识别
# ══════════════════════════════════════════════════════════════

def _llm_detect_columns(df: pd.DataFrame, llm) -> Optional[dict]:
    """用 LLM 识别列角色，返回 meta dict；失败返回 None。"""
    try:
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            uniq = df[col].dropna().unique()[:5]
            samples = [str(v)[:40] for v in uniq]
            lines.append(f"  - {col}  (type={dtype}, 样例={samples})")
        prompt = f"""分析以下 DataFrame 各列的角色。

共 {len(df.columns)} 列：
{chr(10).join(lines)}

对每一列给出角色（只返回 JSON，不要其他文字）：
{{
  "列名": "time / sales / qty / price / category / numeric / ignore"
}}

规则：
- time: 日期/时间列
- sales: 销售额、金额、收入等货币类
- qty: 数量、订单量等计数类
- price: 单价、均价
- category: 分类、文本类（用于分组）
- numeric: 其他数值列（ID、编码等）
- ignore: 应忽略的列
"""
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        import json, re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None
        result = json.loads(m.group())
        meta = {"sales_cols": [], "qty_cols": [], "price_cols": [],
                "cat_cols": [], "numeric_cols": [], "time_cols": []}
        for col, role in result.items():
            if col not in df.columns:
                continue
            if role == "time":
                meta["time_cols"].append(col)
            elif role == "sales":
                meta["sales_cols"].append(col)
                meta["numeric_cols"].append(col)
            elif role == "qty":
                meta["qty_cols"].append(col)
                meta["numeric_cols"].append(col)
            elif role == "price":
                meta["price_cols"].append(col)
                meta["numeric_cols"].append(col)
            elif role == "numeric":
                meta["numeric_cols"].append(col)
            elif role == "category":
                meta["cat_cols"].append(col)
        # 对 LLM 漏掉的角色，用启发式补充
        heur = _detect_columns(df)
        for k in meta:
            if not meta[k] and heur.get(k):
                meta[k] = heur[k]
        return meta
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  安全求和（强制转数值再 sum，绝对不会拼接字符串）
# ══════════════════════════════════════════════════════════════

def _safe_sum(series) -> float:
    """对任意列安全求数值和，字符串→NaN→忽略。"""
    s = pd.to_numeric(series, errors="coerce")
    return float(s.sum()) if s.notna().any() else 0.0


# ══════════════════════════════════════════════════════════════
#  报告生成主函数
# ══════════════════════════════════════════════════════════════

def generate_full_report(
    df: pd.DataFrame,
    file_name: str = "",
    time_col: str = "",
    llm=None,
) -> dict:
    # LLM 列识别（优先），失败则回退启发式
    meta = _llm_detect_columns(df, llm) if llm else None
    if not meta:
        meta = _detect_columns(df)

    charts: list = []
    steps: list = []
    all_findings_parts: list = []

    # ── 将所有识别出的数值列强制转换为纯数值，防止混入的字符串导致 .sum() 拼接 ──
    for col in set(meta["numeric_cols"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 自动识别时间粒度 ──
    granularity = _detect_granularity(df, time_col) if time_col else "M"

    # ════════════════════════════════════════════════
    #  1. 数据概览
    # ════════════════════════════════════════════════
    rows = len(df)
    period = _detect_period(df, time_col) if time_col else ""
    period_short = _detect_period_short(df, time_col) if time_col else ""
    overview = f"## 数据概览\n\n"
    if period:
        overview += f"- **报告周期**：{period}\n"
    if file_name:
        overview += f"- **数据来源**：{file_name}\n"
    overview += f"- **数据行数**：{rows} 条\n"
    overview += f"- **数据列数**：{len(df.columns)} 列\n"
    col_debug = (f"销售额列={meta['sales_cols']}, 数量列={meta['qty_cols']}, "
                 f"分类列={meta['cat_cols']}, 时间列={meta['time_cols']}")
    steps.append({"step": "数据概览", "detail": f"{rows}行 × {len(df.columns)}列 | {col_debug}"})

    # ════════════════════════════════════════════════
    #  2. 核心指标卡
    # ════════════════════════════════════════════════
    kpi_lines = []
    kpi_text = ""

    sales_col = meta["sales_cols"][0] if meta["sales_cols"] else (
        meta["numeric_cols"][0] if meta["numeric_cols"] else None
    )
    total_sales = None
    if sales_col:
        try:
            _s = _safe_sum(df[sales_col])
            total_sales = round(_s, 2)
        except Exception:
            total_sales = None
        if total_sales is not None:
            kpi_lines.append(f"| 总销售额 | **{total_sales:,.2f}** |")
            kpi_text += f"总销售额 {total_sales:,.2f}"

    qty_col = meta["qty_cols"][0] if meta["qty_cols"] else None
    total_qty = None
    if qty_col:
        try:
            _q = _safe_sum(df[qty_col])
            total_qty = int(_q)
        except Exception:
            total_qty = None
        if total_qty is not None:
            kpi_lines.append(f"| 总订单量 | **{total_qty:,}** |")
            kpi_text += f"，总订单量 {total_qty:,}"

    avg_price = None
    if total_sales and total_qty and total_qty > 0:
        avg_price = round(total_sales / total_qty, 2)
        kpi_lines.append(f"| 平均客单价 | **{avg_price:,.2f}** |")
        kpi_text += f"，平均客单价 {avg_price:,.2f}"

    # 环比
    mom_text = ""
    if time_col and total_sales is not None:
        mom = _compute_mom(df, time_col, sales_col, granularity)
        if mom is not None:
            arrow = "↑" if mom > 0 else "↓" if mom < 0 else "→"
            kpi_lines.append(f"| 环比变化 | **{arrow} {abs(mom):.1f}%** |")
            mom_text = f"，环比{mom:+.1f}%"

    # 同比
    yoy_text = ""
    if time_col and total_sales is not None:
        yoy = _compute_yoy(df, time_col, sales_col)
        if yoy is not None:
            arrow_y = "↑" if yoy > 0 else "↓" if yoy < 0 else "→"
            kpi_lines.append(f"| 同比变化 | **{arrow_y} {abs(yoy):.1f}%** |")
            yoy_text = f"，同比{yoy:+.1f}%"

    if kpi_lines:
        kpi_section = "## 核心指标\n\n| 指标 | 数值 |\n|------|------|\n" + "\n".join(kpi_lines) + "\n"
    else:
        kpi_section = ""

    # 一句话结论（由 AI 生成，填入概览末尾）
    one_liner = ""
    if llm:
        one_liner = _generate_one_liner(kpi_text, mom_text, yoy_text, period_short)
    if one_liner:
        overview += f"\n> {one_liner}\n"

    steps.append({"step": "核心指标", "detail": kpi_text[:100] if kpi_text else "已计算"})

    # ════════════════════════════════════════════════
    #  3. 趋势可视化
    # ════════════════════════════════════════════════
    if time_col and sales_col and meta["numeric_cols"]:
        trend_charts = _build_trend_charts(df, time_col, sales_col, granularity)
        charts.extend(trend_charts["charts"])
        steps.extend(trend_charts["steps"])
        if trend_charts.get("outlier_text"):
            all_findings_parts.append(trend_charts["outlier_text"])

    # ════════════════════════════════════════════════
    #  4. 分类对比图
    # ════════════════════════════════════════════════
    cat_col = meta["cat_cols"][0] if meta["cat_cols"] else None
    cat_text_parts = []
    if cat_col and sales_col:
        # 分类柱状图
        top = df.groupby(cat_col)[sales_col].apply(_safe_sum).sort_values(ascending=False)
        img = _bar_chart(top.index.tolist(), top.values.tolist(), cat_col, sales_col)
        if img:
            charts.append(img)
            steps.append({"step": "分类对比", "detail": f"{cat_col} × {sales_col} Top10"})

        # 分类文字分析（top3 / bottom3）
        if len(top) >= 2:
            top3 = top.head(3)
            bot3 = top.tail(3)
            cat_text_parts.append(f"**表现最好的3个{cat_col}**：")
            for i, (k, v) in enumerate(top3.items(), 1):
                pct = v / top.sum() * 100
                cat_text_parts.append(f"{i}. {k}（{v:,.0f}，占比{pct:.1f}%）")
            if len(top) >= 4:
                cat_text_parts.append(f"\n**表现最差的3个{cat_col}**：")
                for i, (k, v) in enumerate(bot3.items(), 1):
                    pct = v / top.sum() * 100
                    cat_text_parts.append(f"{i}. {k}（{v:,.0f}，占比{pct:.1f}%）")

    # ════════════════════════════════════════════════
    #  5. 关键发现（AI 增强）
    # ════════════════════════════════════════════════
    findings = ""
    if llm and (
        one_liner or kpi_text or mom_text or yoy_text or cat_text_parts or all_findings_parts
    ):
        findings = _generate_findings(
            df, meta, kpi_text, mom_text, yoy_text,
            time_col, period_short, "\n".join(cat_text_parts),
            "\n".join(all_findings_parts), llm,
        )
        if findings:
            steps.append({"step": "AI洞察", "detail": findings[:120]})

    # ════════════════════════════════════════════════
    #  拼接最终回复
    # ════════════════════════════════════════════════
    reply = overview + "\n" + kpi_section

    if cat_text_parts:
        reply += f"\n## 分类分析\n\n" + "\n".join(cat_text_parts) + "\n"

    if findings:
        reply += f"\n## 关键发现\n\n{findings}\n"

    return {
        "reply": reply.strip(),
        "charts": charts,
        "process_steps": steps,
    }


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def _detect_period(df: pd.DataFrame, time_col: str) -> str:
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


def _detect_period_short(df: pd.DataFrame, time_col: str) -> str:
    """返回最新周期的简短描述（如 '2026年3月'）。"""
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if len(dt) == 0:
            return ""
        latest = dt.max()
        return latest.strftime("%Y年%-m月")
    except Exception:
        return ""


def _detect_granularity(df: pd.DataFrame, time_col: str) -> str:
    """自动检测时间粒度：D=日, W=周, M=月。"""
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if len(dt) < 2:
            return "M"
        span = (dt.max() - dt.min()).days
        n = len(dt)
        if span <= 60 and n >= 10:
            return "D"
        elif span <= 180:
            return "W"
        return "M"
    except Exception:
        return "M"


def _get_granularity_label(g: str) -> str:
    return {"D": "日", "W": "周", "M": "月"}.get(g, "月")


def _compute_mom(df: pd.DataFrame, time_col: str, value_col: str, granularity: str = "M") -> Optional[float]:
    """计算环比变化（最近两期对比）。"""
    try:
        freq = granularity if granularity != "D" else "W"
        chart_df = df.copy()
        chart_df["_p"] = pd.to_datetime(chart_df[time_col], errors="coerce").dt.to_period(freq).astype(str)
        grouped = chart_df.groupby("_p")[value_col].apply(_safe_sum).sort_index()
        if len(grouped) < 2:
            return None
        curr = grouped.iloc[-1]
        prev = grouped.iloc[-2]
        if prev == 0:
            return None
        return round((curr - prev) / prev * 100, 1)
    except Exception:
        return None


def _compute_yoy(df: pd.DataFrame, time_col: str, value_col: str) -> Optional[float]:
    """计算同比变化（对比去年同期）。"""
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        df_temp = df.copy()
        df_temp["_year"] = dt.dt.year
        df_temp["_month"] = dt.dt.month
        monthly = df_temp.groupby(["_year", "_month"])[value_col].apply(_safe_sum).reset_index()
        monthly = monthly.sort_values(["_year", "_month"])
        if len(monthly) < 13:
            return None
        latest = monthly.iloc[-1]
        same_month_last_year = monthly[
            (monthly["_month"] == latest["_month"]) &
            (monthly["_year"] == latest["_year"] - 1)
        ]
        if same_month_last_year.empty:
            return None
        prev_val = same_month_last_year.iloc[0][value_col]
        curr_val = latest[value_col]
        if prev_val == 0:
            return None
        return round((curr_val - prev_val) / prev_val * 100, 1)
    except Exception:
        return None


def _generate_one_liner(kpi_text: str, mom_text: str, yoy_text: str, period: str) -> str:
    """直接拼出一句话结论，不调用 LLM。"""
    parts = []
    if kpi_text:
        parts.append(kpi_text)
    if mom_text:
        parts.append(mom_text)
    if yoy_text:
        parts.append(yoy_text)
    if parts:
        prefix = f"**{period or '本期'}核心结论**：" if period else "**核心结论**："
        return prefix + "，".join(parts) + "。"
    return ""


# ══════════════════════════════════════════════════════════════
#  趋势图表构建
# ══════════════════════════════════════════════════════════════

def _build_trend_charts(df: pd.DataFrame, time_col: str,
                        sales_col: str, granularity: str) -> dict:
    """构建趋势图表，返回 {"charts": [], "steps": [], "outlier_text": ""}。"""
    result: dict = {"charts": [], "steps": [], "outlier_text": ""}
    freq_map = {"D": "D", "W": "W", "M": "M"}
    freq = freq_map.get(granularity, "M")
    label = _get_granularity_label(granularity)

    periods = sorted(set(pd.to_datetime(df[time_col], errors="coerce")
                         .dt.to_period(freq).dropna().astype(str)))
    if len(periods) < 2:
        return result

    chart_df = df.copy()
    chart_df["_p"] = pd.to_datetime(chart_df[time_col], errors="coerce").dt.to_period(freq).astype(str)
    grouped = chart_df.groupby("_p")[sales_col].apply(_safe_sum).reset_index()

    # 异常检测（Z-score > 2 视为异常）
    vals = grouped[sales_col].values
    mean_v = np.mean(vals)
    std_v = np.std(vals) if np.std(vals) > 0 else 1
    z_scores = np.abs((vals - mean_v) / std_v)
    outlier_mask = z_scores > 2
    outlier_periods = grouped["_p"][outlier_mask].tolist() if outlier_mask.any() else []
    outlier_vals = vals[outlier_mask].tolist() if outlier_mask.any() else []

    img = _trend_chart(
        grouped["_p"].tolist(), vals.tolist(), sales_col,
        granularity_label=label,
        outlier_periods=outlier_periods, outlier_vals=outlier_vals,
    )
    if img:
        result["charts"].append(img)
        n = len(periods)
        result["steps"].append({"step": "趋势图", "detail": f"{sales_col} {label}度趋势（{n}期）"})

    if outlier_periods:
        outlier_text = (
            f"**异常波动检测**：发现 {len(outlier_periods)} 个异常期"
            f"（{'、'.join(outlier_periods[:5])}{'等' if len(outlier_periods) > 5 else ''}）"
            f"，数值显著偏离均值。"
        )
        result["outlier_text"] = outlier_text

    return result


# ══════════════════════════════════════════════════════════════
#  AI 关键发现（增强版）
# ══════════════════════════════════════════════════════════════

def _generate_findings(df, meta, kpi_text, mom_text, yoy_text,
                       time_col, period_short, cat_text, outlier_text, llm) -> str:
    prompt = f"""你是资深数据分析师。根据以下数据信息，生成一份精简的关键发现（200字以内）。

数据规模：{len(df)} 行 × {len(df.columns)} 列
列名：{', '.join(df.columns[:12])}
时间列：{time_col or '未识别'}
报告周期：{period_short or '未识别'}

核心指标：{kpi_text or '无'}
{'环比：' + mom_text if mom_text else ''}
{'同比：' + yoy_text if yoy_text else ''}

{('分类分析：\n' + cat_text) if cat_text else ''}
{('异常波动：\n' + outlier_text) if outlier_text else ''}

请按以下结构输出，如果某项数据不足以支持分析则跳过该项：

1. **核心结论**（一句话概括本期表现）
2. **亮点与短板**（表现最好/最差的维度，如有分类数据则列出具体品类或区域）
3. **异常说明**（如有异常波动则分析可能原因）
4. **建议方向**（给出1-2条可执行的建议）"""

    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
#  图表模板
# ══════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def _trend_chart(x: list, y: list, ylabel: str, granularity_label: str = "月",
                 outlier_periods: list = None, outlier_vals: list = None) -> Optional[str]:
    """趋势折线图（含最高/最低标注 + 异常点标注）。"""
    try:
        _set_font()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(x)), y, "o-", color="#2563EB", linewidth=2.5, markersize=7)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} {granularity_label}度趋势", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 最高点、最低点
        max_i = np.argmax(y)
        min_i = np.argmin(y)
        ax.annotate(f"最高 {y[max_i]:,.0f}", (max_i, y[max_i]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color="#DC2626", fontweight="bold")
        ax.annotate(f"最低 {y[min_i]:,.0f}", (min_i, y[min_i]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, color="#059669", fontweight="bold")

        # 每个数据点标注
        for i, v in enumerate(y):
            ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#1E40AF")

        # 异常点突出显示
        if outlier_periods and outlier_vals:
            outlier_indices = [x.index(p) for p in outlier_periods if p in x]
            for oi in outlier_indices:
                ax.plot(oi, y[oi], "o", color="#DC2626", markersize=12, zorder=5,
                        markeredgecolor="#991B1B", markeredgewidth=2)

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
