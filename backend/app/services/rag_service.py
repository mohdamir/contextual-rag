from typing import List, Optional, Dict
from llama_index.core.schema import Document
from app.core.hybrid_retrieval import HybridRetrievalSystem
from app.core.vectordb import VectorDB
from app.core.bm25_engine import BM25SearchEngine
from app.core.document_processor import DocumentProcessor
from app.core.llms import embedding_model
import os

class RAGService:
    def __init__(self):
        # Initialize components
        self.vector_db = VectorDB(
            connection_string=os.getenv("DB_CONNECTION_STRING"),
            table_name="document_vectors"
        )
        
        self.ir_engine = BM25SearchEngine()
        self.ir_engine.load()  # Load existing index if available
        
        self.hybrid_retriever = HybridRetrievalSystem(
            vector_db=self.vector_db,
            ir_engine=self.ir_engine,
            fusion_method="rrf"
        )
        
        self.document_processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=64
        )
    
    async def ingest_documents(self, documents: List[Document]) -> None:
        """Process and index documents"""
        processed_docs = self.document_processor.process_documents(documents)
        self.hybrid_retriever.index_documents(processed_docs)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: Optional[str] = None,
        vector_weight: Optional[float] = None
    ) -> List[Dict]:
        """Retrieve relevant documents using hybrid approach"""
        return self.hybrid_retriever.retrieve(
            query,
            top_k=top_k,
            fusion_method=fusion_method,
            vector_weight=vector_weight
        )
    
    async def query(
        self,
        question: str,
        context: Optional[Dict] = None,
        top_k: int = 5
    ) -> str:
        """Full RAG pipeline: retrieve + generate"""
        # Retrieve relevant chunks
        retrieved = await self.retrieve(question, top_k=top_k)
        
        # Prepare context for LLM
        context_str = "\n\n".join([
            f"Document {i+1}:\n{doc['document'].text}"
            for i, doc in enumerate(retrieved)
        ])
        
        # TODO: Integrate with LLM generation
        # This would use Ollama or another LLM to generate the final answer
        # based on the retrieved context
        
        return {
            "answer": "Generated answer based on context",  # Placeholder
            "context": context_str,
            "retrieved_documents": retrieved
        }