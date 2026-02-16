from fastapi import APIRouter, Query
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
        SystemMessage(content=SYSTEM_PROMPT),  # 系统提示
        HumanMessage(content=message),  # 用户消息
    ]

    response = await model.ainvoke(messages)

    return {"reply": response.content}
