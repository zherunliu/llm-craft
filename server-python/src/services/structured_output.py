import json
import re
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.chat_model import get_chat_model_service


class ReportSection(BaseModel):
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容")


class Report(BaseModel):
    title: str = Field(description="报告标题")
    summary: str = Field(description="报告摘要，100字以内")
    sections: list[ReportSection] = Field(description="报告章节列表")
    conclusion: str = Field(description="结论")


class CodeReview(BaseModel):
    score: int = Field(description="代码质量评分，1-10分")
    issues: list[str] = Field(description="发现的问题列表")
    suggestions: list[str] = Field(description="改进建议列表")
    summary: str = Field(description="总体评价")


def extract_json(text: str) -> dict:
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 中的内容
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到 { } 包裹的内容
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"failed to extract JSON: {text[:200]}...")


class StructuredOutputService:
    """不完全支持 with_structured_output()，使用提示词引导输出"""

    def __init__(self):
        self.chat_service = get_chat_model_service()

    async def generate_report(self, topic: str) -> Report:
        model = self.chat_service.get_chat_model()

        # 构建 JSON Schema 描述
        json_schema = """{
    "title": "报告标题",
    "summary": "报告摘要，100字以内",
    "sections": [
        {"title": "章节1标题", "content": "章节1内容"},
        {"title": "章节2标题", "content": "章节2内容"}
    ],
    "conclusion": "结论"
}"""

        messages = [
            SystemMessage(
                content=f"""你是一个专业的技术报告撰写专家。
请根据主题生成一份技术报告，必须严格按照以下 JSON 格式输出，不要输出任何其他内容：
{json_schema}"""
            ),
            HumanMessage(content=f"请生成一份关于「{topic}」的技术报告，只输出 JSON"),
        ]

        response = await model.ainvoke(messages)
        data = extract_json(str(response.content))
        return Report(**data)

    async def review_code(self, code: str, language: str = "python") -> CodeReview:
        model = self.chat_service.get_chat_model()

        json_schema = """{
    "score": 8,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "summary": "总体评价"
}"""

        messages = [
            SystemMessage(
                content=f"""你是一个 {language} 代码审查专家。
请审查代码并按以下 JSON 格式输出评价，不要输出任何其他内容：
{json_schema}"""
            ),
            HumanMessage(
                content=f"审查以下代码，只输出 JSON：\n```{language}\n{code}\n```"
            ),
        ]

        response = await model.ainvoke(messages)
        data = extract_json(str(response.content))
        return CodeReview(**data)


_structured_service: StructuredOutputService | None = None


def get_structured_service() -> StructuredOutputService:
    global _structured_service
    if _structured_service is None:
        _structured_service = StructuredOutputService()
    return _structured_service
