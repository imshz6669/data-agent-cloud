"""
自动导出模块（Task 3）。
- ZIP 打包：Markdown + 图表 PNG 文件（非 base64 内嵌，体积小）
- PDF 可选（需 weasyprint）
"""
import io
import re
import base64
import zipfile
from datetime import datetime
from typing import List, Dict, Optional


def create_zip_export(messages: List[Dict], file_name: str = "") -> bytes:
    """将对话消息导出为 ZIP 文件。

    ZIP 内容：
        report.md          — Markdown 报告（相对路径引用图片）
        charts/
            chart_01.png   — 第一张图表
            chart_02.png   — 第二张图表
            ...

    Args:
        messages: [{"role": "user/assistant", "content": str, "charts": [...]}]
        file_name: 数据文件名（可选）

    Returns:
        ZIP 文件 bytes
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── 收集所有图表，写入 ZIP ──
        all_charts: list = []
        chart_idx = 0

        for msg in messages:
            charts = msg.get("charts", [])
            if not charts:
                chart_data = msg.get("chart")
                if chart_data:
                    charts = [chart_data]
            for ch in charts:
                chart_idx += 1
                png_bytes = _decode_chart(ch)
                if png_bytes:
                    fname = f"charts/chart_{chart_idx:02d}.png"
                    zf.writestr(fname, png_bytes)
                    all_charts.append((chart_idx, fname))

        # ── 生成 Markdown ──
        md = _build_markdown(messages, all_charts, file_name)
        zf.writestr("report.md", md.encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()


def _decode_chart(data_url: str) -> Optional[bytes]:
    """将 data:image/png;base64,... 解码为 bytes。"""
    if not data_url:
        return None
    if data_url.startswith("data:"):
        b64 = re.sub(r"^data:image/\w+;base64,", "", data_url)
    else:
        b64 = data_url
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _build_markdown(messages: list, charts: list, file_name: str) -> str:
    """生成 Markdown 报告（图片用相对路径引用）。"""
    md = "# Data Agent 分析报告\n\n"
    md += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    if file_name:
        md += f"数据文件：{file_name}\n\n"
    md += "---\n\n"

    # 建立图表索引：第几张图 → 文件名
    chart_map = {idx: fname for idx, fname in charts}

    q_num = 0
    chart_seq = 0
    for msg in messages:
        if msg["role"] == "user":
            q_num += 1
            md += f"## 💬 问题 {q_num}\n\n{msg['content']}\n\n"
        else:
            md += f"### 🤖 分析结果\n\n{msg['content']}\n\n"

            # 图片引用
            n_charts = len(msg.get("charts", []))
            if n_charts == 0 and msg.get("chart"):
                n_charts = 1

            for _ in range(n_charts):
                chart_seq += 1
                if chart_seq in chart_map:
                    md += f"![图表 {chart_seq}]({chart_map[chart_seq]})\n\n"

            steps = msg.get("process_steps", [])
            if steps:
                md += "<details>\n<summary>🔍 分析过程</summary>\n\n"
                for ps in steps:
                    md += f"- **{ps['step']}**\n  ```\n  {ps['detail']}\n  ```\n"
                md += "\n</details>\n\n"

        md += "---\n\n"

    return md


# ══════════════════════════════════════════════════════════════
#  PDF 导出（可选，需 weasyprint）
# ══════════════════════════════════════════════════════════════

def create_pdf_export(messages: List[Dict], file_name: str = "") -> Optional[bytes]:
    """将对话消息导出为 PDF（需 weasyprint）。

    返回 PDF bytes，如果 weasyprint 不可用则返回 None。
    """
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        return None

    # 构建简单 HTML
    html = _build_html(messages, file_name)
    try:
        pdf_buf = io.BytesIO()
        HTML(string=html).write_pdf(pdf_buf)
        pdf_buf.seek(0)
        return pdf_buf.getvalue()
    except Exception:
        return None


def _build_html(messages: list, file_name: str) -> str:
    """构建 PDF 用 HTML。"""
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'WenQuanYi Zen Hei', sans-serif; max-width: 800px; margin: auto; }",
        "h1 { color: #1E40AF; border-bottom: 2px solid #2563EB; }",
        "h2 { color: #2563EB; }",
        "img { max-width: 100%; page-break-inside: avoid; }",
        "@page { margin: 2cm; }",
        "</style></head><body>",
        f"<h1>Data Agent 分析报告</h1>",
        f"<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    if file_name:
        parts.append(f"<p>数据文件：{file_name}</p>")
    parts.append("<hr>")

    for msg in messages:
        role = "💬 用户" if msg["role"] == "user" else "🤖 助手"
        parts.append(f"<h2>{role}</h2>")
        parts.append(f"<p>{msg['content']}</p>")

        charts = msg.get("charts", [])
        if not charts:
            c = msg.get("chart")
            if c:
                charts = [c]
        for ch in charts:
            parts.append(f'<img src="{ch}" style="max-width:100%">')

        parts.append("<hr>")

    parts.append("</body></html>")
    return "\n".join(parts)
