from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.config import get_settings
from api import metadata, predictions


settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metadata.router)
app.include_router(predictions.router)

try:
    from api.chatbot_router import router as chatbot_router

    app.include_router(chatbot_router, prefix="/api")
except Exception:
    # The prediction API must remain available even when Gemini/chatbot deps are absent.
    chatbot_router = None


STATIC_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

