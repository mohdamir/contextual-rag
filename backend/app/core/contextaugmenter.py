from app.core.vectordb import VectorDB, IRSearchEngine
from typing import List, Dict
from llama_index.core import Document

class ContextAugmenter:
    def __init__(
        self
    ):
        self.context_prompt = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """
    
    @staticmethod
    def _default_embedding_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def index_documents(self, documents: List[Document]):
        """Index documents in both vector and IR systems"""
        # Index for IR search
        self.ir_engine.index_documents(documents)
        
        # Generate embeddings and index for vector search
        texts = [doc.text for doc in documents]
        ids = [doc.doc_id for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        self.vector_db.index_vectors(ids, embeddings, documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",  # "rrf" or "simple"
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """Retrieve documents using hybrid approach"""
        # Vector search
        query_embedding = self.embedding_model.encode(query)
        vector_results = self.vector_db.search_vectors(query_embedding, top_k)
        
        # IR search
        ir_results = self.ir_engine.search(query, top_k)
        
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
                'ir_rank': None
            }
        
        for rank_list in ranked_lists:
            for doc_id, rank in rank_list.items():
                scores[doc_id]['score'] += 1.0 / (k + rank + 1)
                if rank_list is ranked_lists[0]:
                    scores[doc_id]['vector_rank'] = rank + 1
                else:
                    scores[doc_id]['ir_rank'] = rank + 1
        
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
                'ir_rank': r['ir_rank']
            }
        } for r in sorted_results]
