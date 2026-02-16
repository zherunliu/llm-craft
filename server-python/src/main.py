from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.ai import router as ai_router

load_dotenv()

app = FastAPI(
    title="LLM-craft server",
    description="server with FastAPI for llm-craft",
    version="1.0.0",
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
