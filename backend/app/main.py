from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional

from backend.app.api.http_routes import router as http_routes_router
from backend.app.api.ws_routes import router as ws_routes_router
from backend.app.services.assistant_service import AssistantService
from backend.app.core.config import Settings, get_settings_instance

assistant_service_instance: Optional[AssistantService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant_service_instance
    print("Starting up AssistantService...")
    settings = get_settings_instance()
    assistant_service_instance = AssistantService(settings=settings)
    await assistant_service_instance.startup()
    print("AssistantService started.")
    yield
    print("Shutting down AssistantService...")
    if assistant_service_instance:
        await assistant_service_instance.shutdown()
        print("AssistantService shut down.")
    assistant_service_instance = None # Clear the instance

app = FastAPI(lifespan=lifespan, title="FastRTC Voice Assistant Backend")

app.include_router(http_routes_router)
app.include_router(ws_routes_router)

@app.get("/")
async def root():
    return {"message": "FastRTC Voice Assistant Backend is running"}