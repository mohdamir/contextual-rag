import os
import time
import numpy as np
import pandas as pd
import json
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics import ndcg_score
from app.api.query import perform_query, get_hybrid_retriever
from app.core.hybridretriever import HybridRetrievalSystem
from app.models.schemas import QueryRequest, GroundTruthItem
from .utils import load_ground_truth_files
from app.core.llms import llm, embedding_model
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
import torch

from dotenv import load_dotenv
load_dotenv()

# Initialize models
similarity_model = embedding_model
GT_DIR = "./data/ground_truth"

def calculate_similarity(answer1: str, answer2: str) -> float:
    """Calculate cosine similarity between two answers"""
    answer1_embedding = similarity_model.get_text_embedding(answer1)
    answer2_embedding = similarity_model.get_text_embedding(answer2)
    
    if not isinstance(answer1_embedding, torch.Tensor):
        answer1_embedding = torch.tensor(answer1_embedding)
    if not isinstance(answer2_embedding, torch.Tensor):
        answer2_embedding = torch.tensor(answer2_embedding)
    
    if len(answer1_embedding.shape) == 1:
        answer1_embedding = answer1_embedding.unsqueeze(0)
    if len(answer2_embedding.shape) == 1:
        answer2_embedding = answer2_embedding.unsqueeze(0)
    
    return util.cos_sim(answer1_embedding, answer2_embedding).item()

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
        retriever = get_hybrid_retriever()
        retrieved_chunks = retriever.retrieve(query_request.query, top_k=top_k, fusion_method="rrf")
        response = perform_query(query_request, retrieved_chunks, llm)
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
    
    return metrics