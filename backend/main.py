import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.model_loader import registry
    from backend.schemas import ContentRequest, HybridRequest, PredictionResponse, PropagationRequest, HybridPredictionResponse
except ImportError:  # pragma: no cover - direct execution from backend directory
    from model_loader import registry
    from schemas import ContentRequest, HybridRequest, PredictionResponse, PropagationRequest, HybridPredictionResponse


app = FastAPI(title="Fake News & Misinformation Detection API", version="1.0.0")

origins = [
    os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    registry.load_all_models()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "models": registry.load_all_models()}


@app.post("/predict/content", response_model=PredictionResponse)
def predict_content(payload: ContentRequest) -> Dict[str, Any]:
    try:
        return registry.predict_content(payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/propagation", response_model=PredictionResponse)
def predict_propagation(payload: PropagationRequest) -> Dict[str, Any]:
    try:
        return registry.predict_propagation(payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/hybrid", response_model=HybridPredictionResponse)
def predict_hybrid(payload: HybridRequest) -> Dict[str, Any]:
    try:
        return registry.predict_hybrid(payload.text, payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/all")
def predict_all(payload: HybridRequest) -> Dict[str, Any]:
    try:
        content_result = registry.predict_content(payload.text)
        propagation_result = registry.predict_propagation(payload.model_dump())
        hybrid_result = registry.predict_hybrid(payload.text, payload.model_dump())
        return {
            "content": content_result,
            "propagation": propagation_result,
            "hybrid": hybrid_result,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
