import os
from typing import Dict, List
from app.core.llms import get_ollama_llm
from app.models.schemas import QueryResponse, QueryResponseSource
from llama_index.core.llms import ChatMessage
from app.core.hybridretriever import HybridRetrievalSystem
from app.core.llms import get_ollama_llm
from app.services.crew_service import CrewService, CrewAIConfig
from app.core.bm25engine import BM25TFIDFEngine, BMI25_STORE_PATH
from app.core.vectordb import PGVectorDB

import json
import time
from phoenix.otel import register
from dotenv import load_dotenv
load_dotenv()

POSTGRES_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))


def get_rag_service():
    return RagService()

def get_hybrid_retriever() -> HybridRetrievalSystem:
    vector_db = PGVectorDB(POSTGRES_URL, table_name="contextual_rag", embed_dim=CHUNK_SIZE, recreate_table=False)
    ir_engine = BM25TFIDFEngine.load_from_disk(persist_dir=BMI25_STORE_PATH)
    return HybridRetrievalSystem(
        vector_db=vector_db,
        ir_engine=ir_engine
    )

retriever = get_hybrid_retriever()
crew_service = CrewService(config=CrewAIConfig(verbose=True, max_iter=1))

class RagService:
    def __init__(self):
        pass

    def chat(self, query:str, top_k:int):
        retrieval_prompt = query
        llm_prompt = query
        try:
            response = crew_service.create_prompt_enhancer_crew(query)
            prompt_data = json.loads(response)
            retrieval_prompt = prompt_data["retrieval_prompt"]
            llm_prompt = prompt_data["llm_prompt"]
        except Exception as e:
            print (f"Prompt enhancement failed {e}")

        print(f"Optimized Query: {llm_prompt}")

        try: 
            top_k_before_reranking = 2 * top_k
            results = retriever.retrieve(retrieval_prompt, top_k=top_k_before_reranking, fusion_method="rrf")
            print(f"Retrieved {len(results)} unranked documents")

            re_ranked_results = retriever.rerank_and_score_documents(results)
            top_scored_results = sorted(re_ranked_results, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

            print(f"Retrieved {len(top_scored_results)} re-ranked documents")
            if len(top_scored_results) < 1:
                return QueryResponse(answer="No relevant documents found.", sources=[], latency=0.0)
            else:
                return self.perform_query(llm_prompt, top_scored_results)
            
        except Exception as e:
            print(f"Failed to get answer of question: {e}")
            return QueryResponse(answer="No relevant documents found.", sources=[], latency=0.0)
        
    def perform_query(self, prompt: str, retrieved_chunks: List[Dict]) -> QueryResponse:
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
            f"Question: {prompt}\n"
            f"Answer:"
        )

        # Instantiate your LLM
        llm = get_ollama_llm()

        # Build the chat history
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt)
        ]

        # Call the LLM directly
        response = llm.chat(messages)
        response_text = response.message.content

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