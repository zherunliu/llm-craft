from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.services.chat_model import get_chat_model_service
from src.services.memory import get_memory_service
from src.services.rag import get_rag_service
from src.services.guardrail import get_guardrail
from src.services.tools import ALL_TOOLS
from src.services.structured_output import get_structured_service, Report, CodeReview

router = APIRouter(prefix="/ai", tags=["AI"])

SYSTEM_PROMPT = """
You are a programming expert. Your name is Raina. You help users solve programming problems, with a focus on three areas:
- Planning programming learning paths
- Providing programming study advice
- Sharing high-frequency interview questions
Please solve users' programming problems using professional language.
"""


async def build_rag_context(query: str) -> str:
    rag_service = get_rag_service()

    results = await rag_service.retrieve_with_score(query, k=3, score_threshold=0.3)

    if not results:
        print(f"[RAG] no relevant documents found for query: {query}")
        return ""

    print(f"[RAG] found {len(results)} relevant documents")

    # 拼接检索到的文档内容
    context_parts = [
        "here are some relevant documents that might help answer the question:"
    ]
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"\n--- 文档 {i} (similarity: {score:.2f}, source: {source}) ---\n{doc.page_content}"
        )

    return "\n".join(context_parts)


@router.get("/chat/sync")
# Query(...) 必填
async def chat_sync(
    message: str = Query(..., description="用户消息"),
    memory_id: str = Query("default", description="会话ID，用于区分不同对话"),
):
    guardrail = get_guardrail()
    check_result = guardrail.validate(message)
    if not check_result.safe:
        raise HTTPException(
            status_code=400,
            detail=f"input validation failed: {'; '.join(check_result.failures)}",
        )

    chat_service = get_chat_model_service()
    model = chat_service.get_chat_model()

    memory_service = get_memory_service()
    history = memory_service.get_history(memory_id)

    rag_context = await build_rag_context(message)

    enhanced_prompt = SYSTEM_PROMPT
    if rag_context:
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\n{rag_context}"

    messages = [
        SystemMessage(content=enhanced_prompt),
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
    guardrail = get_guardrail()
    check_result = guardrail.validate(message)
    if not check_result.safe:
        # 流式接口返回错误信息
        async def error_stream():
            yield f"data: [error] {'; '.join(check_result.failures)}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    chat_service = get_chat_model_service()
    streaming_model = chat_service.get_streaming_model()

    memory_service = get_memory_service()
    history = memory_service.get_history(memory_id)

    rag_context = await build_rag_context(message)

    enhanced_prompt = SYSTEM_PROMPT
    if rag_context:
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\n{rag_context}"

    messages = [
        SystemMessage(content=enhanced_prompt),
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


@router.get("/chat/tools")
async def chat_with_tools(
    message: str = Query(..., description="用户消息"),
):
    chat_service = get_chat_model_service()
    model = chat_service.get_chat_model()

    model_with_tools = model.bind_tools(ALL_TOOLS)

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
            + "\n\nyou can use the following tools: "
            + ", ".join(t.name for t in ALL_TOOLS)
        ),
        HumanMessage(content=message),
    ]

    response = await model_with_tools.ainvoke(messages)

    if response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"[Tool] invoke: {tool_name}, args: {tool_args}")

            tool_func = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_func:
                result = tool_func.invoke(tool_args)
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "result": result,
                    }
                )

        messages.append(response)
        for tr in tool_results:
            messages.append(
                ToolMessage(content=tr["result"], tool_call_id=tr["tool_call_id"])
            )

        final_response = await model.ainvoke(messages)
        return {
            "reply": final_response.content,
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]} for tc in response.tool_calls
            ],
        }

    return {"reply": response.content, "tool_calls": []}


@router.get("/chat/report")
async def generate_report(
    topic: str = Query(..., description="报告主题"),
) -> Report:
    structured_service = get_structured_service()
    report = await structured_service.generate_report(topic)
    return report


@router.post("/chat/code-review")
async def review_code(
    code: str = Query(..., description="要审查的代码"),
    language: str = Query("python", description="编程语言"),
) -> CodeReview:
    structured_service = get_structured_service()
    review = await structured_service.review_code(code, language)
    return review
