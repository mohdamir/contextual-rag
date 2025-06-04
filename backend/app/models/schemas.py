from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponseSource(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[QueryResponseSource]

class GroundTruthItem(BaseModel):
    question: str
    answer: str
    page: int
    span_start: int
    span_end: int
    context: str  # Added context for recall calculation

class GroundTruthFile(BaseModel):
    filename: str
    items: List[GroundTruthItem]

class EvaluationRequest(BaseModel):
    top_k: int = 3

class EvaluationReport(BaseModel):
    latency: float
    similarity_score: float
    recall_at_k: float
    details: Dict[str, Any] = {}