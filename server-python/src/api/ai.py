from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.chat_model import get_chat_model_service
from src.services.memory import get_memory_service

router = APIRouter(prefix="/ai", tags=["AI"])

SYSTEM_PROMPT = """
You are a programming expert. Your name is Raina. You help users solve programming problems, with a focus on three areas:
- Planning programming learning paths
- Providing programming study advice
- Sharing high-frequency interview questions
Please solve users' programming problems using professional language.
"""


@router.get("/chat/sync")
# Query(...) 必填
async def chat_sync(
    message: str = Query(..., description="用户消息"),
    memory_id: str = Query("default", description="会话ID，用于区分不同对话"),
):
    chat_service = get_chat_model_service()
    model = chat_service.get_chat_model()

    memory_service = get_memory_service()
    history = memory_service.get_history(memory_id)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=message),
    ]

    memory_service.add_user_message(memory_id, message)

    response = await model.ainvoke(messages)

    memory_service.add_ai_message(memory_id, str(response.content))

    return {"reply": response.content}


@router.get("/chat")
async def chat_stream(
    message: str = Query(..., description="用户消息"),
    memory_id: str = Query("default", description="会话ID，用于区分不同对话"),
):
    chat_service = get_chat_model_service()
    streaming_model = chat_service.get_streaming_model()

    memory_service = get_memory_service()
    history = memory_service.get_history(memory_id)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=message),
    ]

    memory_service.add_user_message(memory_id, message)

    async def generate():
        full_response = ""

        async for chunk in streaming_model.astream(messages):
            if chunk.content:
                # 按 SSE 格式发送：data: 内容\n\n
                full_response += str(chunk.content)
                yield f"data: {chunk.content}\n\n"

        memory_service.add_ai_message(memory_id, full_response)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # for Server-Sent Events (SSE)
        headers={
            "Cache-Control": "no-cache",  # 禁用缓存
            "Connection": "keep-alive",  # 保持连接
            "Access-Control-Allow-Origin": "*",  # 允许跨域
        },
    )
