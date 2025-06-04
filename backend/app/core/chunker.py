import os
from dotenv import load_dotenv
from typing import List
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from app.core.llms import embedding_model

from dotenv import load_dotenv
load_dotenv()

class PDFChunkerBase:
    """Base class for PDF chunking strategies."""
    def parse(self, file_path: str) -> List[Document]:
        raise NotImplementedError("Derived classes must implement parse()")
    
    def get_documents(self, file_path: str) -> List[Document]:
        """Helper method to load documents from a file path."""
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        # Optionally, add filename to metadata
        for doc in documents:
            doc.metadata["filename"] = os.path.basename(file_path)
        return documents

class OverlapChunker(PDFChunkerBase):
    """High Overlap Chunking: fixed chunk size with high overlap."""
    def __init__(self, chunk_size=512, chunk_overlap=256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str) -> List[Document]:
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        documents = self.get_documents(file_path)
        return documents
        
class SemanticChunker(PDFChunkerBase):
    """Context-Enriched/Sliding Window Chunking using SlideNodeParser."""
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def parse(self, file_path: str) -> List[Document]:
        # Load documents
        documents = self.get_documents(file_path)
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embedding_model)
        base_splitter = SentenceSplitter(chunk_size=self.chunk_size)
        nodes = splitter.get_nodes_from_documents(documents)

        return [Document(text=node.text, metadata=node.metadata) for node in nodes]

# Dependency injection factory
def get_chunker_from_env() -> PDFChunkerBase:
    strategy = os.getenv("CHUNKING_STRATEGY", "high_overlap").lower()
    chunk_size = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 40))
    if strategy == "sematic_chunking":
        return SemanticChunker(chunk_size=chunk_size, window_size=window_size)
    return OverlapChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
