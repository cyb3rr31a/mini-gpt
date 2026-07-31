import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router, batching_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the batching engine when the server boots
    worker_task = asyncio.create_task(batching_worker())
    yield
    worker_task.cancel()

# Initialize the app and attach the router
app = FastAPI(lifespan=lifespan)
app.include_router(router)