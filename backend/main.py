import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
import json as json_module
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# 改成相对导入（注意前面加点）
from .tools import parse_uploaded_file, DATA_STORE
from .agent import process_chat

# ---------- FastAPI 应用 ----------
app = FastAPI(title="数据分析智能体后端")

# 允许跨域（Streamlit 前端在另一个端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    thread_id: str
    message: str
    file_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    chart: Optional[str] = None
    process_steps: list = []

# ---------- 接口 ----------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件（支持 CSV / Excel / Word），返回文件ID和预览"""
    try:
        content = await file.read()
        df = parse_uploaded_file(content, file.filename)
        file_id = str(uuid.uuid4())
        DATA_STORE[file_id] = df
        return {
            "file_id": file_id,
            "filename": file.filename,
            "preview": df.head(5).to_dict(orient="records"),
            "columns": list(df.columns),
            "shape": list(df.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """对话接口，接收用户消息并返回Agent的回复"""
    # 将 file_id 拼接到消息中，方便Agent识别当前使用的文件
    if req.file_id:
        user_message = f"[当前文件ID: {req.file_id}] {req.message}"
    else:
        user_message = req.message

    try:
        result = process_chat(req.thread_id, user_message)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads")
async def get_threads():
    """获取所有会话列表（thread_id + 消息数 + 第一条用户消息预览）"""
    from .agent import agent, DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    )
    threads = []
    for (tid,) in cur.fetchall():
        config = {"configurable": {"thread_id": tid}}
        state = agent.get_state(config)
        messages = state.values.get("messages", []) if state and state.values else []
        # 取第一条用户消息作为预览
        preview = ""
        msg_count = 0
        for m in messages:
            if m.type in ("human", "ai"):
                msg_count += 1
                if m.type == "human" and not preview:
                    preview = m.content[:50]
        threads.append({
            "thread_id": tid,
            "message_count": msg_count,
            "preview": preview
        })
    conn.close()
    # 按 thread_id 倒序（新的在前）
    threads.sort(key=lambda x: x["thread_id"], reverse=True)
    return {"threads": threads}


@app.get("/history")
async def get_history(thread_id: str):
    """获取指定会话的历史消息"""
    from .agent import agent
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
    except Exception as e:
        return {"history": [], "error": str(e)}

    if not state or not state.values:
        return {"history": []}

    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        # 兼容多种消息类型表示方式
        msg_type = getattr(msg, "type", None)
        if msg_type is None:
            # 回退：尝试通过类名判断
            cls_name = type(msg).__name__
            if "Human" in cls_name:
                msg_type = "human"
            elif "AI" in cls_name or "Assistant" in cls_name:
                msg_type = "ai"
            else:
                continue

        if msg_type in ("human", "ai"):
            content = getattr(msg, "content", "") or ""
            history.append({"role": msg_type, "content": str(content)})
    return {"history": history}


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """删除指定会话的所有数据"""
    from .agent import DB_PATH
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "thread_id": thread_id}