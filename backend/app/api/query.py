import os
import time
from fastapi import APIRouter, Depends
from app.models.schemas import QueryRequest, QueryResponse, QueryResponseSource
from app.core.vectordb import BM25TFIDFEngine, BMI25_STORE_PATH, PGVectorDB
from app.core.hybridretriever import HybridRetrievalSystem
from app.core.llms import llm
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))

router = APIRouter()

def get_hybrid_retriever() -> HybridRetrievalSystem:
    vector_db = PGVectorDB(POSTGRES_URL, table_name="contextual_rag", embed_dim=CHUNK_SIZE, recreate_table=False)
    ir_engine = BM25TFIDFEngine.load_from_disk(persist_dir=BMI25_STORE_PATH)
    return HybridRetrievalSystem(
        vector_db=vector_db,
        ir_engine=ir_engine
    )

def perform_query(request: QueryRequest, retrieved_chunks: List[Dict], llm) -> QueryResponse:
    """Core query function used by both API and evaluator"""
    start_time = time.perf_counter()

    # Building context
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[{i+1}] {chunk['document'].text}\n"

    # Prompt construction
    prompt = (
        f"Use the following context to answer the user's question as accurately as possible.\n\n"
        f"Context:\n{context}\n"
        f"Question: {request.query}\n"
        f"Answer:"
    )

    # Calling LLM with the constructed prompt
    response = llm.complete(prompt, max_tokens=1024)
    response_text = str(response)

    # Prepare Sources
    sources = []
    for chunk in retrieved_chunks:
        sources.append(QueryResponseSource(
            text=chunk['document'].text,
            metadata=chunk['document'].metadata,
            score=chunk.get('details', {})
        ))

    latency = time.perf_counter() - start_time

    return QueryResponse(
        answer=response_text,
        sources=sources,
        latency=latency
    )


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest, retriever: HybridRetrievalSystem = Depends(get_hybrid_retriever)):

    results = retriever.retrieve(request.query, top_k=request.top_k, fusion_method="rrf")
    print(f"Retrieved {len(results)} documents for query: {request.query}")
    print (f"Results: {results}")
    if not results:
        return QueryResponse(answer="No relevant documents found.", sources=[], latency=0.0)

    return perform_query(request, results, llm)