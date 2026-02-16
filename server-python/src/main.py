from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.ai import router as ai_router
from src.services.rag import get_rag_service

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    FastAPI 的 lifespan 用于在应用启动/关闭时执行代码
    - yield 之前：启动时执行（初始化）
    - yield 之后：关闭时执行（清理）
    """
    print("Starting server...")
    rag_service = get_rag_service()
    await rag_service.init()

    yield  # 应用运行中

    print("Shutting down...")


app = FastAPI(
    title="LLM-craft server",
    description="server with FastAPI for llm-craft",
    version="1.0.0",
    lifespan=lifespan,  # 注册生命周期
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 ai 路由
app.include_router(ai_router)


@app.get("/")
async def root():
    return {"message": "LLM-craft Server with FastAPI is running!", "status": "ok"}


@app.get("/hello")
async def hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}


if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    # reload=True 代码修改后自动重启
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
    )
