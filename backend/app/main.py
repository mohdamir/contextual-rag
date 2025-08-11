import asyncio
import os


# Force the default event loop policy (not uvloop) so nest_asyncio can patch it
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api import ingest, query, evaluate, ground_truth
from starlette.middleware.base import BaseHTTPMiddleware
import json
import re

import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
px.launch_app()

# ✅ Let Phoenix use default gRPC endpoint (localhost:4317)
os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)  # remove manual override
os.environ.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)         # use default (gRPC)

tracer_provider = register(
    project_name="Contextual-RAG",
    auto_instrument=True,
)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

app = FastAPI(
    title="Contextual RAG System",
    description="API for document ingestion, querying, and evaluation",
    version="1.0.0"
)

class SanitizeJSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-type") == "application/json":
            body_bytes = await request.body()
            try:
                body_text = body_bytes.decode("utf-8")
                # Clean control characters
                sanitized_text = re.sub(r'[\x00-\x1f\x7f]', ' ', body_text)
                request._body = sanitized_text.encode("utf-8")
            except Exception as e:
                return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})
        return await call_next(request)


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

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "RAG Evaluation System is running"}