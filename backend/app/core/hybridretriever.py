
import os
from app.core.vectordb import VectorDB, IRSearchEngine
from app.core.llms import embedding_model
from typing import List, Dict
from llama_index.core import Document
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from dotenv import load_dotenv
import numpy as np
load_dotenv()

class HybridRetrievalSystem:
    def __init__(
        self,
        vector_db: VectorDB,
        ir_engine: IRSearchEngine
    ):
        self.vector_db = vector_db
        self.ir_engine = ir_engine

    
    def index_documents(self, documents: List[Document]):
        """Index documents in both vector and IR systems"""
        # Index for IR search
        self.ir_engine.index_documents(documents)
        self.ir_engine.persist()
        
        # Generate embeddings and index for vector search
        texts = [doc.text for doc in documents]
        ids = [doc.doc_id for doc in documents]
        embeddings = []
        for text in texts:
            print (f"Generating embedding for text: {text[:50]}...")  # Print first 50 chars
            embedding = embedding_model.get_text_embedding(text)
            print (f"Embedding: {embedding}")
            embeddings.append(embedding)

        print (f"Indexing {len(documents)} documents in vector DB")
        
        embeddings_np = np.array(embeddings).astype('float32')
    
        print(f"Indexing {len(documents)} documents in vector DB")
        self.vector_db.index_vectors(ids, embeddings_np, documents)
        self.vector_db.persist()

    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",  # "rrf" or "simple"
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """Retrieve documents using hybrid approach"""
        # Vector search
        query_embedding = embedding_model.get_text_embedding(query)
        embeddings_np = np.array(query_embedding).astype('float32')
        vector_results = self.vector_db.search_vectors(embeddings_np, top_k)
        print(f"Vector search results: {vector_results}")
        
        # IR search
        ir_results = self.ir_engine.search(query, top_k)
        print(f"IR search results: {ir_results}")
        
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
            {r['document'].doc_id: i for i, r in enumerate(vector_results)},
            {r['document'].doc_id: i for i, r in enumerate(ir_results)}
        ]
        
        scores = {}
        for doc in vector_results + ir_results:
            doc_id = doc['document'].doc_id
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
