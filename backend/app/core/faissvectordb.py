from app.core.vectordb import VectorDB
import os
import faiss
import numpy as np
from typing import List, Dict, Optional
from llama_index.core import Document
import pickle

VECTOR_STORE_PATH = "./data/faiss_vector_store/index.faiss"

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
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            return vectors
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors
    
    def _validate_index(self):
        if self.index is None:
            raise ValueError("FAISS index not initialized")
    
    def index_vectors(self, ids: List[str], vectors: np.ndarray, documents: List[Document]):
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
        self._validate_index()
        faiss.write_index(self.index, self.persist_file_path)

        docs_path = self.persist_file_path + ".docs"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load_from_disk(cls, file_path: str, dimension: int = 1024):
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
        return self.document_map.get(doc_id)
