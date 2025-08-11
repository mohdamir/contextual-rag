from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.core.evaluator import RAGEvaluator
from app.models.schemas import EvaluationRequest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import math
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("./data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Creates folder if it 

def sanitize_for_json(value):
    """Recursively sanitize dict/list values to remove NaN/Inf floats."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None  # or 0.0 if you prefer numbers
    return value

def fmt(val):
    if isinstance(val, list):
        # Flatten to a single numeric
        val = val[0] if len(val) == 1 else (sum([v for v in val if isinstance(v, (int, float))]) / len(val))
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def generate_pdf_report(filename, results):
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Contextual RAG Evaluation Report</b>", styles['Title']))
    story.append(Spacer(1, 12))

    # Aggregates table
    story.append(Paragraph("<b>Aggregate Metrics</b>", styles['Heading2']))
    agg_data = [["Metric", "Score"]]
    for k, v in results["aggregates"].items():
        agg_data.append([k.replace("_", " ").title(), fmt(v)])
    agg_table = Table(agg_data, hAlign="LEFT")
    agg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(agg_table)
    story.append(Spacer(1, 12))

    # Per query details
    story.append(Paragraph("<b>Per-Query Details</b>", styles['Heading2']))

    for i, detail in enumerate(results["details"], 1):
        story.append(Paragraph(f"<b>Query {i}</b>", styles['Heading3']))
        story.append(Paragraph(f"<b>Question:</b> {detail['question']}", styles['Normal']))
        story.append(Paragraph(f"<b>Ground Truth:</b> {detail['ground_truth']}", styles['Normal']))
        story.append(Paragraph(f"<b>RAG Answer:</b> {detail['rag_answer']}", styles['Normal']))
        
        # Scores Table
        score_data = [
            ["Metric", "Value"],
            ["Similarity", fmt(detail["similarity"])],
            ["Faithfulness", fmt(detail["faithfulness"])],
            ["Answer Relevancy", fmt(detail["answer_relevancy"])],
            ["Context Recall", fmt(detail["context_recall"])],
            ["Context Precision", fmt(detail["context_precision"])],
            ["Latency (s)", fmt(detail["latency"])]
        ]
        score_table = Table(score_data, hAlign="LEFT")
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        story.append(score_table)

        story.append(Paragraph("<b>Sources:</b>", styles['Normal']))
        for s in detail["sources"]:
            story.append(Paragraph(f"- {s}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Build PDF
    doc = SimpleDocTemplate(filename, pagesize=A4)
    doc.build(story)
    print(f"PDF report saved to {filename}")

router = APIRouter()

@router.post("/run")
async def evaluate_performance(request: EvaluationRequest):
    evaulator = RAGEvaluator()
    metrics = evaulator.evaluate_rag(top_k=request.top_k)
    metrics = sanitize_for_json(metrics)
    return metrics


@router.post("/report")
async def evaluate_performance(request: EvaluationRequest):
    evaulator = RAGEvaluator()
    metrics = evaulator.evaluate_rag(top_k=request.top_k)
    filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = str(REPORTS_DIR / filename)
    generate_pdf_report(file_path, metrics)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )
