from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.chat_model import get_chat_model_service

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
async def chat_sync(message: str = Query(..., description="用户消息")):
    chat_service = get_chat_model_service()
    model = chat_service.get_chat_model()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]

    response = await model.ainvoke(messages)

    return {"reply": response.content}


@router.get("/chat")
async def chat_stream(message: str = Query(..., description="用户消息")):
    async def generate():
        chat_service = get_chat_model_service()
        streaming_model = chat_service.get_streaming_model()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]

        async for chunk in streaming_model.astream(messages):
            if chunk.content:
                # 按 SSE 格式发送：data: 内容\n\n
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # for Server-Sent Events (SSE)
        headers={
            "Cache-Control": "no-cache",  # 禁用缓存
            "Connection": "keep-alive",  # 保持连接
            "Access-Control-Allow-Origin": "*",  # 允许跨域
        },
    )
