"""
维度自动识别与过滤（Task 1）。
- 用 pd.to_datetime + PeriodIndex 替代手写正则，覆盖更多日期格式
- 自动探测分类列（城市、渠道等）作为候选维度
- 纯函数，无 Streamlit 依赖；缓存由 app.py 的 @st.cache_data 处理
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ─── 维度提取 ───────────────────────────────────────────────

def extract_dimensions(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """从 DataFrame 提取时间维度和分类维度。

    Returns:
        {"time": [{"column": "日期", "periods": ["2024-01", "2024-02", ...]}],
         "category": [{"column": "城市", "values": ["北京", "上海", ...]}]}
    """
    result: dict = {"time": [], "category": []}
    if df is None or df.empty:
        return result

    for col in df.columns:
        # ── 尝试时间维度 ──
        periods = _try_time_column(df[col])
        if periods:
            result["time"].append({"column": col, "periods": periods})
            continue  # 已经是时间列，不再检查分类

        # ── 尝试分类维度 ──
        cat_values = _try_category_column(df[col])
        if cat_values:
            result["category"].append({"column": col, "values": cat_values})

    return result


def _try_time_column(series: pd.Series) -> Optional[List[str]]:
    """尝试将列解析为日期，提取 YYYY-MM 格式的周期列表。"""
    # pd.to_datetime 覆盖绝大多数日期格式
    try:
        dt = pd.to_datetime(series, errors="coerce")
    except Exception:
        return None

    valid_ratio = dt.notna().sum() / max(len(series), 1)
    if valid_ratio < 0.7:  # 少于 70% 可解析 → 不算日期列
        return None

    # 提取年月（PeriodIndex 比手写正则健壮得多）
    periods = sorted(set(dt.dt.to_period("M").dropna()))
    if len(periods) < 2 or len(periods) > 500:
        return None

    return [str(p) for p in periods]


def _try_category_column(series: pd.Series) -> Optional[List[str]]:
    """检查列是否适合作分类维度。"""
    # 跳过数值列和 ID 类高基数列
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return None
    nu = series.nunique()
    if nu < 2 or nu > 50:
        return None

    vals = series.dropna().unique()
    # 转换为字符串并排序
    return sorted([str(v) for v in vals])


# ─── 维度过滤 ───────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    time_filters: Dict[str, List[str]],
    category_filters: Dict[str, List[str]],
) -> pd.DataFrame:
    """对 DataFrame 应用维度过过滤。

    Args:
        df: 原始 DataFrame
        time_filters: {"日期": ["2024-01", "2024-02"]}
        category_filters: {"城市": ["北京", "上海"]}
    """
    filtered = df.copy()

    # ── 时间过滤 ──
    for col, selected_periods in time_filters.items():
        if not selected_periods or col not in filtered.columns:
            continue
        try:
            dt = pd.to_datetime(filtered[col], errors="coerce")
            period_series = dt.dt.to_period("M")
            target = [pd.Period(p) for p in selected_periods]
            filtered = filtered[period_series.isin(target)]
        except Exception:
            continue

    # ── 分类过滤 ──
    for col, selected_values in category_filters.items():
        if not selected_values or col not in filtered.columns:
            continue
        filtered = filtered[filtered[col].astype(str).isin(selected_values)]

    return filtered
