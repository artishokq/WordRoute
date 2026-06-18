"""WordRoute FastAPI application."""
import sys
from pathlib import Path

# Ensure backend/ is on the path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.classifier import get_classifier

app = FastAPI(
    title="WordRoute API",
    description="NLP tool for detecting and analyzing lexical borrowings in Russian",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    print("[startup] loading classifier...")
    clf = get_classifier()
    print(f"[startup] ready, trained={clf.trained}")


@app.get("/")
async def root():
    return {"name": "WordRoute API", "version": "1.0.0", "docs": "/docs"}
