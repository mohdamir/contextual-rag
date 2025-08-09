
import os
from app.core.vectordb import VectorDB, IRSearchEngine
from app.core.llms import query_ollama
from typing import List, Dict
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

class HybridRetrievalSystem:
    def __init__(
        self,
        vector_db: VectorDB,
        ir_engine: IRSearchEngine
    ):
        self.vector_db = vector_db
        self.ir_engine = ir_engine

    def index_nodes(self, documents: List[Document], nodes: List[BaseNode]):
        """Index documents in both vector and IR systems"""
        # Index for IR search
        self.ir_engine.index_documents(documents)
        self.ir_engine.persist()
        
        print(f"Indexing {len(documents)} documents in vector DB")
        self.vector_db.index_nodes(documents, nodes)
        self.vector_db.persist()

    def index_file(self, file_path: str):
        self.vector_db.index_file(file_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",  # "rrf" or "simple"
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """Retrieve documents using hybrid approach"""
        vector_results =self.vector_db.retrieve_from_index(query, top_k)
        print(f"Vector search results: {len(vector_results)}")
        
        # IR search
        ir_results = self.ir_engine.search(query, top_k)
        print(f"IR search results: {len(ir_results)}")
        
        # Fuse results
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(vector_results, ir_results, top_k)
        else:
            return self._simple_weighted_fusion(
                vector_results, ir_results, top_k, vector_weight
            )
    
    def _simple_weighted_fusion(
        self,
        vector_results: List[Dict],
        ir_results: List[Dict],
        top_k: int,
        vector_weight: float
    ) -> List[Dict]:
        """Combine results using weighted scores"""
        combined = {}
        
        # Add vector results
        for result in vector_results:
            doc_id = result['document'].doc_id
            combined[doc_id] = {
                'document': result['document'],
                'vector_score': result['score'],
                'ir_score': 0.0,
                'combined_score': result['score'] * vector_weight
            }
        
        # Add IR results
        for result in ir_results:
            doc_id = result['document'].doc_id
            if doc_id in combined:
                combined[doc_id]['ir_score'] = result['score']
                combined[doc_id]['combined_score'] += result['score'] * (1 - vector_weight)
            else:
                combined[doc_id] = {
                    'document': result['document'],
                    'vector_score': 0.0,
                    'ir_score': result['score'],
                    'combined_score': result['score'] * (1 - vector_weight)
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: -x['combined_score']
        )[:top_k]
        
        return [{
            'document': r['document'],
            'score': r['combined_score'],
            'type': 'hybrid',
            'details': {
                'vector_score': r['vector_score'],
                'ir_score': r['ir_score']
            }
        } for r in sorted_results]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        ir_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        ranked_lists = [
            {r['document'].id_: i for i, r in enumerate(vector_results)},
            {r['document'].id_: i for i, r in enumerate(ir_results)}
        ]
        
        scores = {}
        for doc in vector_results + ir_results:
            doc_id = doc['document'].id_
            scores[doc_id] = {
                'document': doc['document'],
                'score': 0.0,
                'vector_rank': None,
                'ir_rank': None,
                'vector_score': 0.0,
                'ir_score': 0.0
            }
        
        for rank_list in ranked_lists:
            for doc_id, rank in rank_list.items():
                scores[doc_id]['score'] += 1.0 / (k + rank + 1)
                if rank_list is ranked_lists[0]:
                    scores[doc_id]['vector_rank'] = rank + 1
                    scores[doc_id]['vector_score'] = vector_results[rank]['score']
                else:
                    scores[doc_id]['ir_rank'] = rank + 1
                    scores[doc_id]['ir_score'] = ir_results[rank]['score']
        
        sorted_results = sorted(
            scores.values(),
            key=lambda x: -x['score']
        )[:top_k]
        
        return [{
            'document': r['document'],
            'score': r['score'],
            'type': 'hybrid_rrf',
            'details': {
                'vector_rank': r['vector_rank'],
                'ir_rank': r['ir_rank'],
                'vector_score': r['vector_score'],
                'ir_score': r['ir_score']
            }
        } for r in sorted_results]


    def rerank_and_score_documents(query: str, docs: list) -> list:

        # Construct user prompt with query and passages (truncated to fit context)
        prompt = f"""
        You are a relevance ranking engine.

        Given a query and {len(docs)} passages, return a JSON array of exactly {len(docs)} numeric scores
        in the SAME order as the passages.
        Each score must be between 1.0 (lowest) and 10.0 (highest).
        Do not explain your reasoning. 
        Do not output anything except the JSON array.

        Query: "{query}"

        Passages:
        """

        for i, doc_dict in enumerate(docs, 1):
            snippet = doc_dict['document'].text.replace("\n", " ").strip()[:500]
            prompt += f"{i}. {snippet}\n"

        prompt += f"\nOutput only the JSON array of exactly {len(docs)} scores, like this:\n[9.2, 8.5, 3.1, 7.0]"

        response = query_ollama(prompt=prompt, system_prompt=None)
        print (response)
        try:
            scores = json.loads(response)
            if not isinstance(scores, list) or len(scores) != len(docs):
                raise ValueError("Unexpected scores format or length")
        except Exception as e:
            print(f"Warning: Failed to parse Llama3 rerank scores from Ollama: {e}")
            scores = [5.0] * len(docs)  # fallback neutral score for all

        updated_docs = []
        for doc_dict, score in zip(docs, scores):
            doc = doc_dict['document']
            existing_meta = dict(doc.metadata) if doc.metadata else {}
            existing_meta['rerank_score'] = float(score)

            updated_doc = Document(text=doc.text, metadata=existing_meta, doc_id=doc.id_)

            # Copy original dict and replace 'document' and add top-level 'rerank_score'
            updated_doc_dict = dict(doc_dict)
            updated_doc_dict['document'] = updated_doc
            updated_doc_dict['rerank_score'] = float(score)
            updated_docs.append(updated_doc_dict)

        return updated_docs
