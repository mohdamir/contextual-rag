import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser
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
    def parse(self, documents = List[Document]) -> List[BaseNode]:
        """Parse documents into chunks using specific strategy"""
        pass
    
    def get_documents(self, file_path: str) -> List[Document]:
        """Helper method to load documents from a file path."""
        reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
        documents = reader.load_data(file_path=file_path)
        for doc in documents:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["file_path"] = file_path
        return documents

class OverlapChunker(PDFChunkerBase):
    """Fixed-size chunks with configurable overlap."""
    def parse(self, documents = List[Document]) -> List[BaseNode]:
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes

class SemanticChunker(PDFChunkerBase):
    """Semantic-aware chunking based on content boundaries."""
    def __init__(self, chunk_size: int = 512, breakpoint_threshold: float = 95):
        super().__init__(chunk_size)
        self.breakpoint_threshold = breakpoint_threshold
    
    def parse(self, documents = List[Document]) -> List[BaseNode]:
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=self.breakpoint_threshold,
            embed_model=embedding_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes

class AnthropicContextualChunker(PDFChunkerBase):
    """Hierarchical chunking with Anthropic-style context augmentation"""
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        augmenter: AnthropicContextAugmenter = AnthropicContextAugmenter()
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.augmenter = augmenter or AnthropicContextAugmenter()
    
    def parse(self, documents = List[Document]) -> List[BaseNode]:
        """Parse document with contextual augmentation"""
        print (f"Parsing {len(documents)} documents")
        if len(documents) > 0:
            WHOLE_DOCUMENT = documents[0].text
            
            #node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            try:
                node_parser = MarkdownNodeParser(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                nodes = node_parser.get_nodes_from_documents(documents)
                
                # Process each chunk with contextual augmentation
                augmented_nodes = self.augmenter.create_contextual_nodes(nodes, whole_document=WHOLE_DOCUMENT)
                return augmented_nodes
            except Exception as e:
                print (f"Exception occured in parsing the document {e}")
                raise e
        else:
            print ("No documents to parse. Please check if file is empty")
            raise e


def get_chunker_from_env() -> PDFChunkerBase:
    """Factory method to create chunker based on environment config."""
    strategy = os.getenv("CHUNKING_STRATEGY", "overlap").lower()
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
        return AnthropicContextualChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:  # Default to overlap strategy
        return OverlapChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

