# Data Agent — 技术文档

## 概述

Data Agent 是一个基于大语言模型的智能数据分析助手。用户上传数据文件后，通过自然语言对话即可完成数据查询、统计分析、图表生成等操作。

## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 后端框架 | FastAPI | 0.138+ |
| AI Agent | LangGraph (create_react_agent) | 1.2+ |
| LLM | DeepSeek v4-pro | — |
| 前端 | Streamlit | 1.58+ |
| 数据库 | SQLite (checkpointer) | — |
| 数据处理 | Pandas + NumPy | — |
| 可视化 | Matplotlib + Seaborn | — |
| Python | 3.13 | — |
| 运行环境 | Windows (PowerShell) | — |

## 项目结构

```
data-agent/
├── backend/
│   ├── __init__.py
│   ├── agent.py          # Agent 定义、工具、对话处理
│   ├── main.py           # FastAPI 应用、API 路由
│   └── tools.py          # 数据解析、SQL 生成、Pandas 执行
├── frontend/
│   ├── app.py            # Streamlit 前端界面
│   └── style.css         # 蓝白主题样式
├── .venv/                # Windows 虚拟环境
├── .venv-linux/          # WSL 虚拟环境（备选）
├── agent_memory_v2.db    # SQLite checkpointer 数据库
├── pyproject.toml        # 项目配置
├── requirements.txt      # 依赖列表
├── TECHNICAL.md          # 本文档
└── ISSUES.md             # 问题记录与解决方案
```

## 架构

```
用户浏览器 (Streamlit :8501)
        │
        │ POST /chat
        ▼
   FastAPI (:8000)
        │
        │ process_chat()
        ▼
   LangGraph Agent
        │
        ├── compress_history()   # 记忆压缩
        ├── agent.invoke()        # LLM 推理 + 工具调用
        │       │
        │       ├── load_data()       # 加载文件预览
        │       ├── query_data()      # 自然语言 → SQL
        │       ├── compute_data()    # Pandas 代码执行
        │       └── create_chart()    # Matplotlib 图表生成
        │
        └── 返回 {reply, chart, process_steps}
```

## 核心模块

### 1. Agent (backend/agent.py)

- 使用 `create_react_agent` 构建 ReAct 风格 Agent
- 自定义 `AgentState`，包含 `chart_spec` 字段（非累积型，每次覆盖）
- 系统提示词强调每次独立处理用户请求，图表类型以当前消息为准

### 2. API 路由 (backend/main.py)

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/upload` | 上传文件（CSV/Excel/Word） |
| POST | `/chat` | 对话交互 |
| GET | `/threads` | 获取所有会话列表 |
| GET | `/history?thread_id=` | 获取指定会话历史 |
| DELETE | `/threads/{thread_id}` | 删除会话 |

### 3. 工具函数 (backend/agent.py)

| 工具 | 参数 | 功能 |
|------|------|------|
| `load_data` | file_id | 加载数据预览（前10行） |
| `query_data` | file_id, question | 自然语言 → SQL 查询 |
| `compute_data` | file_id, code | 执行 Pandas 分析代码 |
| `create_chart` | file_id, chart_description | LLM 生成 matplotlib 代码 → 渲染图表 |

### 4. Chart Spec 机制

- `AgentState` 中定义 `chart_spec: str`（普通字符串，不使用 `add_messages` 累加器）
- 每次 `process_chat()` 从用户消息中提取图表类型，通过 `agent.invoke({"chart_spec": ...})` 传入
- `chart_spec` 值：`"bar"` / `"line"` / `"pie"` / `"scatter"` / `"hist"` / `"box"` / `""`
- 空字符串表示用户未要求画图

### 5. 记忆管理

- 使用 `SqliteSaver` 将对话状态持久化到 `agent_memory_v2.db`
- `compress_history()` 在消息超过 15 条时自动压缩：保留最近 10 条，旧消息用 LLM 生成摘要
- `journal_mode=DELETE`（禁用 WAL，避免 WSL/Windows 跨文件系统问题）

## 前端 (frontend/app.py)

- 蓝白配色数据分析主题
- 侧边栏：文件上传 + 历史会话列表（含删除）
- 主界面：聊天消息 + 分析过程面板（可收起）
- 图表通过 base64 解码后以 bytes 传给 `st.image()`

## 中文字体

模块加载时自动检测可用中文字体（Microsoft YaHei / SimHei / WenQuanYi 等），
并注入到 matplotlib 全局配置。`create_chart` 的代码模板中也强制设置字体。

## 数据库

- 文件：`agent_memory_v2.db`
- 表：`checkpoints`（对话检查点）、`writes`（写入记录）
- journal 模式：DELETE（无 WAL 文件）
- 启动时 `VACUUM` 压缩

## 启动方式

```powershell
# 后端（PowerShell）
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 前端（另开终端）
streamlit run frontend/app.py
```

前端默认访问 `http://localhost:8501`。

## 注意事项

1. 项目运行在 Windows 上，虚拟环境为 `.venv\Scripts\python.exe`
2. WSL 下创建的 `.venv-linux` 仅供兼容测试，生产不使用
3. 文件上传数据存储在内存（`DATA_STORE` 字典），重启后丢失
4. DeepSeek API 超时设为 120 秒，单次调用最多重试 1 次
