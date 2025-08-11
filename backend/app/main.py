import asyncio
import os
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true' 
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)  # remove manual override
os.environ.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)         # use default (gRPC)

try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import ingest, query, evaluate, ground_truth
import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from app.utils.middleware import SanitizeJSONMiddleware
from phoenix.otel import register
px.launch_app()

app = FastAPI(
    title="Contextual RAG System",
    description="API for document ingestion, querying, and evaluation",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#app.add_middleware(SanitizeJSONMiddleware)

app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(ground_truth.router, prefix="/groundtruth", tags=["Ground Truth"])

tracer_provider = register(
    project_name="Contextual-RAG",
    set_global_tracer_provider=False,
    auto_instrument=True,
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "RAG Evaluation System is running"}