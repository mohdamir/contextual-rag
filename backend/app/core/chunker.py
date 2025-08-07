import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from llama_index.core.schema import BaseNode
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from app.core.llms import embedding_model
from app.core.contextaugmenter import AnthropicContextAugmenter
from dotenv import load_dotenv

load_dotenv()


class PDFChunkerBase(ABC):
    """Abstract base class for all chunking strategies."""
    def __init__(self, chunk_size: int = 768, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parse documents into chunks using specific strategy"""
        pass
    
    def get_documents(self, file_path: str) -> List[Document]:
        """Helper method to load documents from a file path."""
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in documents:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["file_path"] = file_path
        return documents

class OverlapChunker(PDFChunkerBase):
    """Fixed-size chunks with configurable overlap."""
    def parse(self, file_path: str) -> List[Document]:
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        documents = self.get_documents(file_path)
        nodes = splitter.get_nodes_from_documents(documents)
        return [Document(text=node.text, metadata=node.metadata) for node in nodes]

class SemanticChunker(PDFChunkerBase):
    """Semantic-aware chunking based on content boundaries."""
    def __init__(self, chunk_size: int = 512, breakpoint_threshold: float = 95):
        super().__init__(chunk_size)
        self.breakpoint_threshold = breakpoint_threshold
    
    def parse(self, file_path: str) -> List[Document]:
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=self.breakpoint_threshold,
            embed_model=embedding_model
        )
        documents = self.get_documents(file_path)
        nodes = splitter.get_nodes_from_documents(documents)
        return [Document(text=node.text, metadata=node.metadata) for node in nodes]

class AnthropicContextualChunker(PDFChunkerBase):
    """Hierarchical chunking with Anthropic-style context augmentation"""
    def __init__(
        self,
        chunk_sizes: List[int] = [2048, 512],
        augmenter: Optional[AnthropicContextAugmenter] = None
    ):
        self.chunk_sizes = chunk_sizes
        self.augmenter = augmenter or AnthropicContextAugmenter()
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse document with contextual augmentation"""
        # Load and parse documents hierarchically
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        
        # Process each chunk with contextual augmentation
        augmented_docs = []
        for node in leaf_nodes:
            parent_text = node.parent_node.text if node.parent_node else ""
            
            # Get contextual information from LLM
            context_data = self.augmenter.augment_chunk(
                chunk_text=node.text,
                parent_text=parent_text
            )
            
            # Create augmented document
            augmented_text = (
                f"Document Context Summary: {context_data['summary']}\n\n"
                f"Key Concepts: {', '.join(context_data['keywords'])}\n\n"
                f"Relation to Document: {context_data['relation']}\n\n"
                f"Original Content:\n{node.text}"
            )
            
            augmented_docs.append(Document(
                text=augmented_text,
                metadata={
                    **node.metadata,
                    "context_summary": context_data["summary"],
                    "keywords": context_data["keywords"],
                    "relation": context_data["relation"],
                    "parent_context": parent_text,
                    "original_text": node.text,
                    "filename": os.path.basename(file_path)
                }
            ))
        
        return augmented_docs

class CompositeChunker(PDFChunkerBase):
    """Combines multiple chunking strategies for optimal results."""
    def __init__(
        self,
        chunkers: List[PDFChunkerBase],
        strategy: str = "sequential"  # "sequential" or "parallel"
    ):
        self.chunkers = chunkers
        self.strategy = strategy
    
    def parse(self, file_path: str) -> List[Document]:
        if self.strategy == "sequential":
            documents = []
            for chunker in self.chunkers:
                documents.extend(chunker.parse(file_path))
            return documents
        else:  # parallel
            # Implement parallel processing if needed
            raise NotImplementedError("Parallel chunking not yet implemented")

def get_chunker_from_env() -> PDFChunkerBase:
    """Factory method to create chunker based on environment config."""
    strategy = os.getenv("CHUNKING_STRATEGY", "composite").lower()
    chunk_size = int(os.getenv("CHUNK_SIZE", 768))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 40))
    breakpoint_threshold = float(os.getenv("SEMANTIC_THRESHOLD", 95))
    
    if strategy == "overlap":
        return OverlapChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy == "semantic":
        return SemanticChunker(
            chunk_size=chunk_size,
            breakpoint_threshold=breakpoint_threshold
        )
    elif strategy == "contextual":
        return AnthropicContextualChunker()
    else:  # Default to composite strategy
        return CompositeChunker([
            OverlapChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            SemanticChunker(chunk_size=chunk_size, breakpoint_threshold=breakpoint_threshold),
            AnthropicContextualChunker()
        ])

