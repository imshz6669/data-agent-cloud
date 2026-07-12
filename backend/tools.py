import io
import pandas as pd
import sqlite3
from docx import Document
from typing import Tuple

# 用于存储上传的 DataFrame，key 为 file_id
DATA_STORE = {}


def parse_uploaded_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """根据文件后缀解析 Word / CSV / Excel 为 DataFrame"""
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_content))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_content))
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(file_content))
        tables = doc.tables
        if not tables:
            raise ValueError("Word 文档中没有表格")
        # 简单取第一个表格
        table = tables[0]
        data = [[cell.text for cell in row.cells] for row in table.rows]
        df = pd.DataFrame(data[1:], columns=data[0])
    else:
        raise ValueError(f"不支持的文件类型: {filename}")
    return df


def execute_nl2sql(df: pd.DataFrame, question: str, llm) -> Tuple[str, pd.DataFrame]:
    """将自然语言转为 SQL 并执行，返回 SQL 语句和结果 DataFrame"""
    # 1) 将 DataFrame 写入内存 SQLite
    conn = sqlite3.connect(':memory:')
    df.to_sql('data', conn, index=False, if_exists='replace')

    # 2) 获取表结构信息
    columns = ', '.join([f"{col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)])
    schema = f"表名: data\n列: {columns}\n示例数据:\n{df.head(3).to_string(index=False)}"

    # 3) 让 LLM 生成 SQL（这里用传入的 llm 对象）
    prompt = f"""你是一个 SQL 专家。请将下面的自然语言问题转换为 SQL 查询语句。
数据库使用的是 SQLite，表结构和示例数据如下：
{schema}

⚠️ SQLite 限制（禁止使用以下语法）：
- 不支持 ROLLUP / CUBE / GROUPING SETS → 需要汇总行请手动 UNION ALL 或分开查询
- 不支持窗口函数外的 OVER 子句中 ORDER BY 与聚合混用
- 不支持 FULL OUTER JOIN，请用 LEFT JOIN + UNION + RIGHT JOIN 替代
- 列名如含中文或特殊字符，务必用双引号包裹

问题: {question}

⚠️ 使用限制：
- 不要使用 LIMIT 或 TOP 限制返回行数，返回所有匹配的数据
- 不要使用 OFFSET 或分页

请只返回 SQL 语句，不要包含其他解释或标记。"""
    response = llm.invoke(prompt)
    sql = response.content.strip()
    # 移除可能的 markdown 代码块标记
    if sql.startswith('```sql'):
        sql = sql[6:]
    if sql.endswith('```'):
        sql = sql[:-3]
    sql = sql.strip()

    # 修复 LLM 常见 SQL 拼写错误
    import re as _re
    sql = _re.sub(r'(?i)(FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|JOIN|ON|AND|OR)(["\w])', r'\1 \2', sql)
    sql = _re.sub(r'(\w)(FROM|WHERE|GROUP\s+BY|ORDER\s+BY)', r'\1 \2', sql)
    sql = _re.sub(r'(?i)\bOR\s+DER\s+BY\b', 'ORDER BY', sql)
    sql = _re.sub(r'(?i)\bGRUP\s+BY\b', 'GROUP BY', sql)

    # 4) 执行 SQL
    try:
        result_df = pd.read_sql_query(sql, conn)
        conn.close()
        return sql, result_df
    except Exception as e:
        conn.close()
        # 把错误喂回 LLM 修正一次
        fix_prompt = f"""SQL 执行出错，请修正：
原始 SQL: {sql}
错误信息: {e}

请只返回修正后的 SQL 语句，不含解释。"""
        fixed = llm.invoke(fix_prompt)
        fixed_sql = fixed.content.strip()
        if fixed_sql.startswith('```sql'):
            fixed_sql = fixed_sql[6:]
        if fixed_sql.endswith('```'):
            fixed_sql = fixed_sql[:-3]
        fixed_sql = fixed_sql.strip()
        # 二次清洗
        fixed_sql = _re.sub(r'(?i)\bOR\s+DER\s+BY\b', 'ORDER BY', fixed_sql)
        fixed_sql = _re.sub(r'(?i)\bGRUP\s+BY\b', 'GROUP BY', fixed_sql)
        conn2 = sqlite3.connect(':memory:')
        df.to_sql('data', conn2, index=False, if_exists='replace')
        result_df = pd.read_sql_query(fixed_sql, conn2)
        conn2.close()
        return fixed_sql, result_df


def analyze_data(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """安全执行 Pandas 分析代码，返回计算结果（DataFrame）"""
    local_vars = {'df': df, 'pd': pd}
    try:
        exec(code, {}, local_vars)
        result = local_vars.get('result', None)
        if result is None:
            # 尝试返回最后赋值的变量
            for key in reversed(list(local_vars.keys())):
                if key not in ('df', 'pd') and isinstance(local_vars[key], pd.DataFrame):
                    return local_vars[key]
            return pd.DataFrame()
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame([result])
    except Exception as e:
        return pd.DataFrame({'错误': [str(e)]})

