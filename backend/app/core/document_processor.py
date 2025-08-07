from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        
        for doc in documents:
            # Parse into nodes (chunks)
            nodes = self.parser.get_nodes_from_documents([doc])
            
            # Convert nodes to documents with proper IDs
            for node in nodes:
                doc_id = self._generate_doc_id(node.text, node.metadata)
                processed_docs.append(Document(
                    text=node.text,
                    metadata=node.metadata,
                    doc_id=doc_id
                ))
        
        return processed_docs
    
    def _generate_doc_id(self, text: str, metadata: dict) -> str:
        """Generate consistent doc ID from content and metadata"""
        content_str = text + str(metadata)
        return hashlib.sha256(content_str.encode()).hexdigest()