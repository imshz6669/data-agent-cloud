# 问题记录与解决方案

> 惯例：每次修复非平凡 bug 后在此记录，包含现象、根因、修复、教训。

---

## #1 对话卡顿、不出现结果

**日期**：2026-06-28

**现象**：前端对话一直转圈，长时间不返回结果。

**根因**：
1. `agent_memory.db` 的 WAL 日志文件损坏（15MB + 4.3MB WAL），SQLite 随机触发 `disk I/O error`
2. 数据库在 Windows 盘（`E:\`），WSL 访问跨文件系统 I/O 性能差
3. 所有对话共用 `thread_id="default"`，88 个检查点堆积在同一线程

**修复**：
1. 新建 `agent_memory_v2.db`，用 `sqlite3.backup()` 从备份恢复数据
2. 强制 `PRAGMA journal_mode=DELETE`，禁用 WAL 模式
3. `frontend/app.py` 中 `thread_id` 改为每次生成 UUID
4. 项目改回 Windows 侧启动（`uvicorn` 在 PowerShell 中运行）

**教训**：SQLite WAL 模式在 WSL/Windows 跨文件系统场景下极易损坏。生产环境要么放 WSL 本地，要么从 Windows 原生启动。

---

## #2 LLM 生成图表代码时 import 链断裂

**日期**：2026-06-28

**现象**：`create_chart` 用 `exec()` 执行 LLM 生成的 matplotlib 代码，报各种 `NameError`。

**根因**：为安全起见，`exec()` 使用了自定义 `safe_builtins`（白名单），只包含十几个常用函数。matplotlib 内部复杂的 import 链需要 `importlib`、`functools`、`collections` 等模块，在白名单中缺失。

**修复**：改用 `builtins.__dict__`（完整内置环境）。由于图表代码由 LLM 生成且只操作数据，安全风险可控。

**教训**：对复杂第三方库的 `exec()` 不要做内置函数白名单，要么完整给，要么不用 `exec()`。

---

## #3 pandas 频率别名废弃导致图表报错

**日期**：2026-06-28

**现象**：`Invalid frequency: M. Please use 'ME' instead.`

**根因**：pandas 2.2+ 废弃了旧频率别名。DeepSeek 训练数据较旧，生成的代码使用 `'M'`（月末）、`'Y'`（年末）。

**修复**：在 `create_chart` 的 LLM 提示词中添加显式警告：`'M'→'ME', 'Q'→'QE', 'Y'/'A'→'YE'`。

**教训**：LLM 的知识截止日期与当前库版本间存在差距时，需要在提示词中显式补丁。

---

## #4 图表中文显示为乱码/方格

**日期**：2026-06-28

**现象**：生成的图表中汉字变成方块。

**根因**：
1. matplotlib 默认字体不支持中文
2. 之前用 `fm._load_fontmanager(try_read_cache=False)` 强制刷缓存，反而可能导致字体扫描失败
3. `exec()` 中的代码模板重复调用 `matplotlib.use('Agg')` 可能重置字体配置

**修复**：
1. 模块加载时显式注册 Windows 中文字体文件（`msyh.ttc`、`simhei.ttf` 等）
2. 去掉强刷缓存的代码，用温和的 `fm.fontManager.addfont()` 逐文件注册
3. 在 LLM 代码模板中显式添加 `plt.rcParams['font.sans-serif']` 设置
4. 从模板中移除重复的 `matplotlib.use('Agg')` 调用

**教训**：matplotlib 字体管理是全局状态，需要在最开始统一配置。`exec()` 中的代码也应该显式设置字体，避免依赖继承的环境。

---

## #5 图表类型不匹配——核心问题

**日期**：2026-06-28（多次迭代）

**现象**：用户要求柱状图却生成折线图，或第一张图的类型决定后续所有图表。

**根因**（三层）：
1. **LLM 自行决定图表类型**：DeepSeek 看了数据特征后自己推断图表类型，忽略用户指令
2. **历史记忆污染**：agent 的 checkpointer 保留了历史对话中的图表类型，新请求受旧记忆影响
3. **图表提取取错了**：`process_chat` 用 `break` 取了第一张图，而 agent 在一次请求中生成了两张图（先按历史画旧的，再按新指令画正确的）

**修复**（迭代过程）：
1. 尝试拆分四个独立工具（`draw_bar_chart` 等）→ agent 仍选错
2. 尝试从用户消息解析图表类型注入指令 → agent 仍忽略
3. 尝试强制 `_detect_chart_type` 注入前缀 → agent 仍选错
4. **最终方案**：LangGraph 自定义 `AgentState`，添加 `chart_spec: str`（非累积字段），在 `process_chat` 中每次提取并传入 `invoke({"chart_spec": ...})`。同时去掉图表提取循环中的 `break`，确保取到最后一张图。

**教训**：
- LLM Agent 对精确指令的遵循度不可靠，需要代码层兜底
- LangGraph State 中非 `Annotated` 字段天然具有覆盖语义，适合传递"当前回合有效"的元数据
- 同一请求中 agent 可能调用多次同名工具，提取结果时必须取最后一次

---

## #6 create_react_agent 要求 remaining_steps 字段

**日期**：2026-06-28

**现象**：`ValueError: Missing required key(s) {'remaining_steps'} in state_schema`

**根因**：`create_react_agent` 内部使用 `remaining_steps` 控制递归深度，自定义 `state_schema` 时必须包含。

**修复**：在 `AgentState(TypedDict)` 中添加 `remaining_steps: int`，调用时传 `"remaining_steps": 20`。

---

## #7 st.image() 无法渲染 data URL

**日期**：2026-06-28

**现象**：后端返回 `data:image/png;base64,...` 格式的图表数据，前端不显示。

**根因**：Streamlit 的 `st.image()` 把字符串参数当作文件路径处理，不认识 data URL 协议。

**修复**：前端添加 `decode_chart()` 函数，将 data URL 解码为 bytes 后传入 `st.image()`。

**教训**：不同框架对 data URL 的支持不一致，应在传入渲染函数前统一解码为二进制。

---

## 惯例

- 每次修复非平凡 bug 后，在此文档追加一条记录
- 格式：`## #N 标题` + 现象/根因/修复/教训
- 文件位置：项目根目录 `ISSUES.md`
