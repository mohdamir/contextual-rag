import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics import ndcg_score
from app.api.query import perform_query
from app.models.schemas import QueryRequest, GroundTruthItem
from .utils import load_ground_truth_files
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from dotenv import load_dotenv
load_dotenv()

# Initialize models
similarity_model = OpenAILikeEmbedding(
            api_base=os.getenv("OPENAI_LIKE_API_BASE"),
            model_name=os.getenv("EMBEDDING_MODEL"),
            api_key=os.getenv("OPENAI_LIKE_API_KEY")
        )
GT_DIR = "./data/ground_truth"

def calculate_similarity(answer1: str, answer2: str) -> float:
    """Calculate cosine similarity between two answers"""
    embeddings = similarity_model.encode([answer1, answer2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def calculate_recall(ground_truth: GroundTruthItem, retrieved_sources: list) -> float:
    """Calculate recall for a single query"""
    # Check if ground truth context appears in any retrieved source
    gt_context = ground_truth.context
    for source in retrieved_sources:
        if gt_context in source.text:
            return 1.0
    return 0.0

def evaluate_rag(top_k: int = 3) -> dict:
    """Run full RAG evaluation using ground truth"""
    gt_data = load_ground_truth_files(GT_DIR)
    if not gt_data:
        return {"similarity": 0.0, "recall@k": 0.0}
    
    metrics = {
        "latencies": [],
        "similarities": [],
        "recalls": [],
        "details": []
    }
    
    for item in gt_data:
        if not isinstance(item, dict):
            continue
            
        gt_item = GroundTruthItem(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            page=item.get("page", 0),
            span_start=item.get("span_start", 0),
            span_end=item.get("span_end", 0),
            context=item.get("context", "")
        )
        
        if not gt_item.question:
            continue
            
        # Run query
        start_time = time.perf_counter()
        query_request = QueryRequest(query=gt_item.question, top_k=top_k)
        response = perform_query(query_request)
        latency = time.perf_counter() - start_time
        
        # Calculate metrics
        similarity = calculate_similarity(gt_item.answer, response.answer)
        recall = calculate_recall(gt_item, response.sources)
        
        # Store results
        metrics["latencies"].append(latency)
        metrics["similarities"].append(similarity)
        metrics["recalls"].append(recall)
        metrics["details"].append({
            "question": gt_item.question,
            "ground_truth": gt_item.answer,
            "rag_answer": response.answer,
            "similarity": similarity,
            "recall": recall,
            "latency": latency,
            "sources": [s.text[:200] + "..." for s in response.sources]
        })
    
    # Calculate aggregate metrics
    return {
        "latency": np.mean(metrics["latencies"]) if metrics["latencies"] else 0.0,
        "similarity": np.mean(metrics["similarities"]) if metrics["similarities"] else 0.0,
        "recall@k": np.mean(metrics["recalls"]) if metrics["recalls"] else 0.0,
        "details": metrics["details"]
    }