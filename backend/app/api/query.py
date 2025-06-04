import os
import time
from fastapi import APIRouter, Depends
from app.models.schemas import QueryRequest, QueryResponse, QueryResponseSource
from app.core.vectordb import FaissVectorDB, BM25TFIDFEngine, VECTOR_STORE_PATH, BMI25_STORE_PATH, DIMENSIONS
from app.core.hybridretriever import HybridRetrievalSystem
from typing import List, Dict
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

def get_hybrid_retriever() -> HybridRetrievalSystem:
    vector_db = FaissVectorDB.load_from_disk(file_path=VECTOR_STORE_PATH, dimension=DIMENSIONS)
    ir_engine = BM25TFIDFEngine.load_from_disk(persist_dir=BMI25_STORE_PATH)
    return HybridRetrievalSystem(
        vector_db=vector_db,
        ir_engine=ir_engine
    )

llm = OpenAILike(
    api_base=os.getenv("OPENAI_LIKE_API_BASE"),
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_LIKE_API_KEY"),
    temperature=0.3,
)

def perform_query(request: QueryRequest, retrieved_chunks: List[Dict], llm) -> QueryResponse:
    """Core query function used by both API and evaluator"""
    start_time = time.perf_counter()

    # Sorting chunks
    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda x: x.get('combined_score', x.get('vector_score', 0)),
        reverse=True
    )
    top_chunks = sorted_chunks[:5]  # Use top 5 chunks as context (adjust as needed)

    # Building context
    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"[{i+1}] {chunk['document'].text}\n"

    # Prompt construction
    prompt = (
        f"Use the following context to answer the user's question as accurately as possible.\n\n"
        f"Context:\n{context}\n"
        f"Question: {request.query}\n"
        f"Answer:"
    )

    # Calling LLM with the constructed prompt
    response_text = llm.generate(prompt)

    # Prepare Sources
    sources = []
    for chunk in top_chunks:
        sources.append(QueryResponseSource(
            text=chunk['document'].text,
            metadata=chunk['document'].metadata,
            score=chunk.get('combined_score', chunk.get('vector_score', 0))
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
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i+1} ({result['type']}, score: {result['score']:.3f}):")
        print(f"Document: {result['document'].metadata.get('filename', 'N/A')}")
        print(f"Text: {result['document'].text[:200]}...")
        if 'details' in result:
            print(f"Details: {result['details']}")

    return perform_query(request, results)