"""FastAPI wrapper around the pipeline.

    python -m uvicorn app:app --reload --port 8000
    -> http://127.0.0.1:8000

Drop your own static/index.html in and it will be served at /.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import vectorstore
from config import BASE_DIR, config
from pipeline import ask

app = FastAPI(title=config.project_name, version="1.0.0")

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    namespace: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    hits: List[Dict[str, Any]]


@app.get("/")
def home():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"service": config.project_name, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Confirms credentials work and reports how many vectors are indexed."""
    try:
        s = vectorstore.stats()
        return {
            "status": "ok",
            "project": config.project_name,
            "dimension": s.get("dimension"),
            "total_vectors": s.get("total_vector_count"),
            "embedding_model": config.embedding_model,
            "llm_models": config.llm_models,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = ask(req.query, top_k=req.top_k, namespace=req.namespace)
    except Exception as e:
        return ChatResponse(answer=f"Request failed: {e}", sources=[], hits=[])
    return ChatResponse(**result)
