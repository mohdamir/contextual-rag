import os
import time
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, QueryResponseSource
from app.core.vectordb import BM25TFIDFEngine, BMI25_STORE_PATH, PGVectorDB
from app.core.hybridretriever import HybridRetrievalSystem
from app.core.llms import llm, query_ollama
from app.services.crew_service import CrewService, CrewAIConfig
from typing import List, Dict

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor 
from phoenix.otel import register

from dotenv import load_dotenv

load_dotenv()

os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)  # remove manual override
os.environ.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)         # use default (gRPC)
tracer_provider = register(
    project_name="Contextual-RAG",
    auto_instrument=True,
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

POSTGRES_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))

router = APIRouter()

crew_service = CrewService(config=CrewAIConfig(verbose=True, max_iter=1, trace_provider=tracer_provider))


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

    system_prompt = """You are an expert AI assistant trained to answer questions strictly based on the provided context. Follow these rules:

        1. **Source-Based Answers**:  
        - Only use information from the given context to formulate answers.  
        - If the context is insufficient, respond: "The document does not contain relevant information."  

        2. **Precision & Clarity**:  
        - Provide concise, well-structured answers.  
        - Use bullet points or numbered lists for multi-part answers when appropriate.  

        3. **Honesty & Transparency**:  
        - Never hallucinate or invent details.  
        - Explicitly state when you’re uncertain due to missing context.  

        4. **Formatting**:  
        - Highlight key terms **like this** for emphasis.  
        - Maintain a neutral, professional tone.  

        5. **User Intent**:  
        - If the question is ambiguous, request clarification while suggesting possible interpretations.  

        Example Interaction:  
        User: "What are the key features of Project X?"  
        Context: [Document describing Project X]  
        You: "Based on the document:  
        - **Feature 1**: [Description]  
        - **Feature 2**: [Description]  
        [Source: Section 3.2 of the document]"
        """

    # Prompt construction
    prompt = (
        f"Use the following context to answer the user's question as accurately as possible.\n\n"
        f"Context:\n{context}\n"
        f"Question: {request.query}\n"
        f"Answer:"
    )

    # Calling LLM with the constructed prompt
    response = query_ollama(prompt=prompt, system_prompt=system_prompt, model="llama3.2:3b")
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

    try:
        optimized = crew_service.create_prompt_enhancer_crew(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    print(f"Optimized Query: {optimized}")

    top_k_before_reranking = 2 * request.top_k
    results = retriever.retrieve(optimized, top_k=top_k_before_reranking, fusion_method="rrf")
    print(f"Retrieved {len(results)} unranked documents")

    re_ranked_results = retriever.rerank_and_score_documents(results)
    top_scored_results = sorted(re_ranked_results, key=lambda x: x['rerank_score'], reverse=True)[:request.top_k]


    print(f"Retrieved {len(top_scored_results)} documents")
    print (f"Results: {top_scored_results}")
    if not top_scored_results:
        return QueryResponse(answer="No relevant documents found.", sources=[], latency=0.0)

    return perform_query(request, top_scored_results, llm)