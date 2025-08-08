from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import ingest, query, evaluate, ground_truth
import os
import phoenix as px
from llama_index.core import set_global_handler
px.launch_app()
#http://localhost:6006

set_global_handler("arize_phoenix")

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

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

app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(ground_truth.router, prefix="/groundtruth", tags=["Ground Truth"])

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "RAG Evaluation System is running"}