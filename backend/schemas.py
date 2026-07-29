from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Article or claim text to analyze")


class PropagationRequest(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    node_features: Optional[List[List[float]]] = None


class HybridRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Article or claim text to analyze")
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    node_features: Optional[List[List[float]]] = None


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class HybridPredictionResponse(PredictionResponse):
    component_scores: Dict[str, Any]
