import os
from functools import lru_cache


class Settings:
    def __init__(self):
        # Ollama 配置
        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "glm-5:cloud")
        self.ollama_embedding_model: str = os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "embeddinggemma"
        )

        # 服务器配置
        self.server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port: int = int(os.getenv("SERVER_PORT", "8000"))


@lru_cache()
def get_settings() -> Settings:
    """
    @lru_cache() 装饰器确保只创建一次 Settings 实例
    这是一种简单的单例模式实现
    """
    return Settings()
