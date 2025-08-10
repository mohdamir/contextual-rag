import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from llama_index.core import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle

BMI25_STORE_PATH = "./data/bm25_index_store"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

class IRSearchEngine(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Document]):
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
        pass

class BM25TFIDFEngine(IRSearchEngine):
    def __init__(self, persist_dir: str = BMI25_STORE_PATH):
        self.bm25 = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.corpus = []
        self.documents = []
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
    
    def _get_index_path(self, index_type: str) -> str:
        """Get file path for saved index"""
        return os.path.join(self.persist_dir, f"{index_type}_index.pkl")
    
    def index_documents(self, documents: List[Document]):
        """Index documents for BM25 and TF-IDF"""
        self.documents = documents
        self.corpus = [doc.text for doc in documents]
        
        # Train BM25
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Train TF-IDF
        self.tfidf.fit(self.corpus)
    
    def persist(self):
        """Save both BM25 and TF-IDF indices to disk"""
        if self.bm25 is None or not hasattr(self.tfidf, 'vocabulary_'):
            raise ValueError("Indices not initialized - call index_documents() first")
        
        # Save BM25 index
        with open(self._get_index_path("bm25"), 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'tokenized_corpus': [doc.split() for doc in self.corpus],
                'documents': self.documents
            }, f)
        
        # Save TF-IDF vectorizer
        with open(self._get_index_path("tfidf"), 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf,
                'corpus': self.corpus,
                'documents': self.documents
            }, f)
    
    @classmethod
    def load_from_disk(cls, persist_dir: str = BMI25_STORE_PATH):
        """Load indices from disk"""
        instance = cls(persist_dir)
        
        # Load BM25
        bm25_path = instance._get_index_path("bm25")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                instance.corpus = bm25_data['corpus']
                instance.documents = bm25_data['documents']
                instance.bm25 = BM25Okapi(bm25_data['tokenized_corpus'])
        
        # Load TF-IDF
        tfidf_path = instance._get_index_path("tfidf")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                tfidf_data = pickle.load(f)
                instance.tfidf = tfidf_data['vectorizer']
                if not hasattr(instance, 'corpus'):
                    instance.corpus = tfidf_data['corpus']
                if not hasattr(instance, 'documents'):
                    instance.documents = tfidf_data['documents']
        
        return instance
    
    def search(self, query: str, top_k: int) -> List[Dict]:
        """Search using both BM25 and TF-IDF, return combined results"""
        if not self.bm25 or not hasattr(self.tfidf, 'vocabulary_'):
            raise ValueError("Indices not initialized - call index_documents() first")
        
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # TF-IDF search
        query_vec = self.tfidf.transform([query])
        doc_vectors = self.tfidf.transform(self.corpus)
        tfidf_scores = (doc_vectors * query_vec.T).toarray().flatten()
        tfidf_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        
        # Combine results using reciprocal rank fusion
        combined_results = {}
        
        # Process BM25 results
        for rank, idx in enumerate(bm25_indices, 1):
            doc = self.documents[idx]
            if doc.doc_id not in combined_results:
                combined_results[doc.doc_id] = {
                    'document': doc,
                    'bm25_score': bm25_scores[idx],
                    'bm25_rank': rank,
                    'tfidf_score': 0,
                    'tfidf_rank': float('inf'),
                    'id_': doc.doc_id
                }
        
        # Process TF-IDF results
        for rank, idx in enumerate(tfidf_indices, 1):
            doc = self.documents[idx]
            if doc.doc_id in combined_results:
                combined_results[doc.doc_id]['tfidf_score'] = tfidf_scores[idx]
                combined_results[doc.doc_id]['tfidf_rank'] = rank
            else:
                combined_results[doc.doc_id] = {
                    'document': doc,
                    'bm25_score': 0,
                    'bm25_rank': float('inf'),
                    'tfidf_score': tfidf_scores[idx],
                    'tfidf_rank': rank,
                    'id_': doc.doc_id
                }
        
        # Calculate combined score using reciprocal rank fusion
        for doc_id in combined_results:
            rrf_score = (1 / (60 + combined_results[doc_id]['bm25_rank']) +
                        1 / (60 + combined_results[doc_id]['tfidf_rank']))
            combined_results[doc_id]['score'] = rrf_score
        
        return sorted(
            combined_results.values(),
            key=lambda x: -x['score']
        )[:top_k]