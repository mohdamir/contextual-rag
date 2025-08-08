from fastapi import APIRouter
from app.core.evaluator import RAGEvaluator
from app.models.schemas import EvaluationRequest, EvaluationReport
import time

router = APIRouter()

@router.post("/")
async def evaluate_performance(request: EvaluationRequest):
    start_time = time.perf_counter()
    evaulator = RAGEvaluator()

    metrics = evaulator.evaluate_rag(top_k=request.top_k)
    latency = time.perf_counter() - start_time

    return metrics
