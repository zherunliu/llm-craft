from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ChatMemoryService:
    def __init__(self, max_messages: int = 10):
        self._store: dict[str, list[BaseMessage]] = {}
        self.max_messages = max_messages

    def get_history(self, memory_id: str) -> list[BaseMessage]:
        return self._store.get(memory_id, [])

    def add_message(self, memory_id: str, message: BaseMessage):
        if memory_id not in self._store:
            self._store[memory_id] = []

        history = self._store[memory_id]
        history.append(message)

        if len(history) > self.max_messages:
            # 保留最新的 max_messages 条
            self._store[memory_id] = history[-self.max_messages :]

    def add_user_message(self, memory_id: str, content: str):
        self.add_message(memory_id, HumanMessage(content=content))

    def add_ai_message(self, memory_id: str, content: str):
        self.add_message(memory_id, AIMessage(content=content))

    def clear_history(self, memory_id: str):
        if memory_id in self._store:
            del self._store[memory_id]

    def get_all_memory_ids(self) -> list[str]:
        return list(self._store.keys())


_memory_service: ChatMemoryService | None = None


def get_memory_service() -> ChatMemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = ChatMemoryService()
    return _memory_service
