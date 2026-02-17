from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from src.core.config import get_settings


class ChatModelService:
    def __init__(self):
        settings = get_settings()

        self.chat_model: BaseChatModel = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

        self.streaming_model: BaseChatModel = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            streaming=True,  # type: ignore
        )

        print(
            f"ChatModel initialized: {settings.ollama_model} at {settings.ollama_base_url}"
        )

    def get_chat_model(self) -> BaseChatModel:
        return self.chat_model

    def get_streaming_model(self) -> BaseChatModel:
        return self.streaming_model


# 创建单例实例
# 在 Python 中，模块级变量天然是单例的
_chat_model_service: ChatModelService | None = None


def get_chat_model_service() -> ChatModelService:
    """
    获取 ChatModelService 单例
    使用延迟初始化，只在第一次调用时创建实例
    """
    global _chat_model_service
    if _chat_model_service is None:
        _chat_model_service = ChatModelService()
    return _chat_model_service
