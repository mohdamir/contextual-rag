from fastapi import APIRouter
from app.core.evaluator import RAGEvaluator
from app.models.schemas import EvaluationRequest

router = APIRouter()

@router.post("/run")
async def evaluate_performance(request: EvaluationRequest):
    evaulator = RAGEvaluator()
    metrics = evaulator.evaluate_rag(top_k=request.top_k)
    return metrics


@router.post("/report")
async def evaluate_performance(request: EvaluationRequest):
    evaulator = RAGEvaluator()
    metrics = evaulator.evaluate_rag(top_k=request.top_k)
    report = evaulator.generate_report(metrics=metrics)
    return report
