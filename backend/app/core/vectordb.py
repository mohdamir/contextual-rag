import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import ensure_directory_exists
import pickle
from pathlib import Path

VECTOR_STORE_PATH = "./data/faiss_vector_store/index.faiss"
BMI25_STORE_PATH = "./data/bm25_index_store"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 1024

# ====================== Abstract Interfaces ======================

class VectorDB(ABC):
    @abstractmethod
    def index_vectors(self, ids: List[str], vectors: np.ndarray, documents: List[Document]):
        pass
    
    @abstractmethod
    def search_vectors(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        pass

class IRSearchEngine(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Document]):
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
        pass

class FaissVectorDB(VectorDB):
    def __init__(self, dimension: int = 1024, persist_file_path: str = VECTOR_STORE_PATH):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.document_map = {}
        self.persist_file_path = persist_file_path
        os.makedirs(os.path.dirname(self.persist_file_path), exist_ok=True)
        self._initialize_faiss_index()
    
    def _initialize_faiss_index(self):
        """Initialize or load FAISS index"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        if len(vectors) == 0:
            return vectors
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors
    
    def _validate_index(self):
        """Ensure index is ready for operations"""
        if self.index is None:
            raise ValueError("FAISS index not initialized")
    
    def index_vectors(self, ids: List[str], vectors: np.ndarray, documents: List[Document]):
        """Index vectors with FAISS"""
        if len(vectors) == 0 or len(documents) == 0:
            print("No vectors or documents to index")
            return
        
        self._validate_index()
        vectors = self._normalize_vectors(vectors)
        
        # Add vectors to index
        self.index.add(vectors)
        
        # Store document references
        for id_, doc in zip(ids, documents):
            self.document_map[id_] = doc
        self.documents.extend(documents)
    
    def search_vectors(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """Search FAISS index"""
        self._validate_index()
        print(f"Index total vectors: {self.index.ntotal}")
        print(f"Query vector shape: {query_vector.shape}")
        print(f"Top_k requested: {top_k}")

        query_vector = self._normalize_vectors(query_vector)
        print(f"Normalized query vector: {query_vector}")

        distances, indices = self.index.search(query_vector, top_k)
        print(f"FAISS search distances: {distances}")
        print(f"FAISS search indices: {indices}")

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                try:
                    doc = self.documents[idx]
                    results.append({
                        'score': float(distances[0][i]),
                        'document': doc,
                        'type': 'vector'
                    })
                except IndexError:
                    print(f"IndexError: idx={idx}, documents={len(self.documents)}")
                    continue

        print(f"Results: {results}")
        return results

    
    def persist(self):
        """Persist FAISS index to disk"""
        
        self._validate_index()
        faiss.write_index(self.index, self.persist_file_path)

        docs_path = self.persist_file_path + ".docs"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load_from_disk(cls, file_path: str, dimension: int = 1024):
        """Load FAISS index from disk"""
        instance = cls(dimension)
        instance.index = faiss.read_index(file_path)
        docs_path = file_path + ".docs"
        try:
            with open(docs_path, "rb") as f:
                instance.documents = pickle.load(f)
            instance.document_map = {doc.doc_id: doc for doc in instance.documents}
        except FileNotFoundError:
            instance.documents = []
            instance.document_map = {}
        return instance

    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by its ID"""
        return self.document_map.get(doc_id)

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
                    'tfidf_rank': float('inf')
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
                    'tfidf_rank': rank
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