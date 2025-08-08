import os
import time
import numpy as np
import pandas as pd
import json
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics import ndcg_score
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from app.api.query import perform_query, get_hybrid_retriever
from app.core.hybridretriever import HybridRetrievalSystem
from app.models.schemas import QueryRequest, GroundTruthItem
from .utils import load_ground_truth_files
from app.core.llms import llm, embedding_model
import torch
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Initialize models
similarity_model = embedding_model
GT_DIR = "./data/ground_truth"

class RAGEvaluator:
    def __init__(self):
        self.llm = llm
        self.embedding_model = embedding_model
        
    def calculate_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate cosine similarity between two answers"""
        answer1_embedding = self.embedding_model.get_text_embedding(answer1)
        answer2_embedding = self.embedding_model.get_text_embedding(answer2)
        
        if not isinstance(answer1_embedding, torch.Tensor):
            answer1_embedding = torch.tensor(answer1_embedding)
        if not isinstance(answer2_embedding, torch.Tensor):
            answer2_embedding = torch.tensor(answer2_embedding)
        
        if len(answer1_embedding.shape) == 1:
            answer1_embedding = answer1_embedding.unsqueeze(0)
        if len(answer2_embedding.shape) == 1:
            answer2_embedding = answer2_embedding.unsqueeze(0)
        
        return util.cos_sim(answer1_embedding, answer2_embedding).item()

    def calculate_recall(self, ground_truth: GroundTruthItem, retrieved_sources: list) -> float:
        """Calculate recall for a single query"""
        gt_context = ground_truth.context
        for source in retrieved_sources:
            if gt_context in source.text:
                return 1.0
        return 0.0

    def evaluate_with_ragas(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Evaluate using RAGAs metrics"""
        result = evaluate(
            dataset={
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            },
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ],
            llm=self.llm
        )
        return {k: v for k, v in result.items()}

    def evaluate_rag(self, top_k: int = 3) -> Dict[str, Any]:
        """Run full RAG evaluation with enhanced metrics"""
        gt_data = load_ground_truth_files(GT_DIR)
        if not gt_data:
            return {"error": "No ground truth data found"}

        metrics = {
            "latencies": [],
            "similarities": [],
            "recalls": [],
            "faithfulness": [],
            "answer_relevancy": [],
            "context_recall": [],
            "context_precision": [],
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
            response = perform_query(query_request, retrieved_chunks, self.llm)
            latency = time.perf_counter() - start_time

            # Calculate basic metrics
            similarity = self.calculate_similarity(gt_item.answer, response.answer)
            recall = self.calculate_recall(gt_item, response.sources)

            # Calculate RAGAs metrics
            ragas_result = self.evaluate_with_ragas(
                question=gt_item.question,
                answer=response.answer,
                contexts=[s.text for s in response.sources]
            )

            # Store results
            metrics["latencies"].append(latency)
            metrics["similarities"].append(similarity)
            metrics["recalls"].append(recall)
            metrics["faithfulness"].append(ragas_result["faithfulness"])
            metrics["answer_relevancy"].append(ragas_result["answer_relevancy"])
            metrics["context_recall"].append(ragas_result["context_recall"])
            metrics["context_precision"].append(ragas_result["context_precision"])
            
            metrics["details"].append({
                "question": gt_item.question,
                "ground_truth": gt_item.answer,
                "rag_answer": response.answer,
                "similarity": similarity,
                "recall": recall,
                "faithfulness": ragas_result["faithfulness"],
                "answer_relevancy": ragas_result["answer_relevancy"],
                "context_recall": ragas_result["context_recall"],
                "context_precision": ragas_result["context_precision"],
                "latency": latency,
                "sources": [s.text[:200] + "..." for s in response.sources]
            })

        # Calculate aggregate statistics
        metrics["aggregates"] = {
            "mean_similarity": np.mean(metrics["similarities"]),
            "mean_recall": np.mean(metrics["recalls"]),
            "mean_faithfulness": np.mean(metrics["faithfulness"]),
            "mean_answer_relevancy": np.mean(metrics["answer_relevancy"]),
            "mean_context_recall": np.mean(metrics["context_recall"]),
            "mean_context_precision": np.mean(metrics["context_precision"]),
            "mean_latency": np.mean(metrics["latencies"])
        }

        return metrics

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        report = f"""
        RAG Evaluation Report
        =====================
        Total Queries Evaluated: {len(metrics['details'])}
        
        Performance Metrics:
        - Average Similarity: {metrics['aggregates']['mean_similarity']:.2f}
        - Average Recall@k: {metrics['aggregates']['mean_recall']:.2f}
        - Average Faithfulness: {metrics['aggregates']['mean_faithfulness']:.2f}
        - Average Answer Relevancy: {metrics['aggregates']['mean_answer_relevancy']:.2f}
        - Average Context Recall: {metrics['aggregates']['mean_context_recall']:.2f}
        - Average Context Precision: {metrics['aggregates']['mean_context_precision']:.2f}
        - Average Latency: {metrics['aggregates']['mean_latency']:.2f}s
        
        Detailed Findings:
        """
        
        for i, detail in enumerate(metrics["details"], 1):
            report += f"""
            Query {i}: {detail['question']}
            - Ground Truth: {detail['ground_truth'][:100]}...
            - RAG Answer: {detail['rag_answer'][:100]}...
            - Similarity: {detail['similarity']:.2f}
            - Recall: {detail['recall']:.2f}
            - Faithfulness: {detail['faithfulness']:.2f}
            - Answer Relevancy: {detail['answer_relevancy']:.2f}
            - Context Recall: {detail['context_recall']:.2f}
            - Context Precision: {detail['context_precision']:.2f}
            - Latency: {detail['latency']:.2f}s
            """
            
        return report