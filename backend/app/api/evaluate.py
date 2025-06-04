from fastapi import APIRouter
from app.core.evaluator import evaluate_rag
from app.models.schemas import EvaluationRequest, EvaluationReport
import time

router = APIRouter()

@router.post("/", response_model=EvaluationReport)
async def evaluate_performance(request: EvaluationRequest):
    start_time = time.perf_counter()
    metrics = evaluate_rag(top_k=request.top_k)
    latency = time.perf_counter() - start_time
    
    return EvaluationReport(
        latency=latency,
        similarity_score=metrics["similarity"],
        recall_at_k=metrics["recall@k"],
        details=metrics.get("details", [])
    )