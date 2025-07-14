import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict

import anyio.to_thread
from fastapi import FastAPI, status

from src.agents.orchestrator import agent_orchestrator
from src.core.config import get_settings
from src.services.qdrant.vector_db import Qdrant

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """Application startup and shutdown events."""
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = settings.CONCURRENT_THREAD_COUNT
    asyncio.create_task(agent_orchestrator())
    async with Qdrant(index_name=settings.QDRANT_COLLECTION):
        ...

    yield


app = FastAPI(docs_url="/", lifespan=lifespan)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="localhost",
        port=10100,
        workers=settings.WORKER_COUNT,
        loop="uvloop" if os.name == "posix" else "asyncio",
    )
