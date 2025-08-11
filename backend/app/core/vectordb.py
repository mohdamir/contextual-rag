import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import Document, NodeWithScore, BaseNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from app.core.llms import get_embedding_model
from llama_index.core import load_index_from_storage
from sqlalchemy import make_url, text
from app.core.logger import logger

# Define storage paths
STORAGE_DIR = "./data/storage"
DOCSTORE_PATH = os.path.join(STORAGE_DIR, "docstore.json")
INDEX_STORE_PATH = os.path.join(STORAGE_DIR, "index_store.json")

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)


# ====================== Abstract Interfaces ======================

class VectorDB(ABC):
    @abstractmethod
    def index_nodes(self, docuemnts: List[Document], nodes: List[BaseNode]) -> bool:
        pass
    
    @abstractmethod
    def retrieve_from_index(self, query_data) -> List[Dict]:
        pass
    
    @abstractmethod
    def delete_by_file_name(self, file_name: str) -> bool:
        pass

class PGVectorDB(VectorDB):
    def __init__(
        self,
        connection_string: str,
        table_name: str = "llama_vector_store",
        embed_dim: int = 768,
        recreate_table: bool = False
    ):
        """
        Initialize PGVector store
        
        Args:
            connection_string: Postgres connection string
                (e.g., "postgresql://user:password@localhost:5432/dbname")
            table_name: Name of the table to store vectors
            embed_dim: Dimension of the embeddings
            recreate_table: Whether to drop and recreate the table
        """
        print(f"Using connection string: {connection_string}")
        self.connection_string = connection_string
        self.table_name = table_name
        self.embed_dim = int(embed_dim)
        self.vector_store = self._init_vector_store(recreate_table)
        self.doc_store = SimpleDocumentStore()
        self.index_store = SimpleIndexStore()

        if recreate_table == False:
            self._load_stores()
        self.document_map = {}

    def _init_vector_store(self, recreate_table: bool) -> PGVectorStore:

        self._ensure_vector_extension()

        """Initialize the PGVector store with proper parameters"""
        if recreate_table:      
            self._drop_table_if_exists()

        url = make_url(self.connection_string)
        return PGVectorStore.from_params(
            database="postgres",  # Fixed database name
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=self.table_name,
            embed_dim=self.embed_dim  # Embedding dimension fetched from MongoDB
        )

    def _drop_table_if_exists(self):
        """Drop the table if it exists"""
        from sqlalchemy import create_engine
        
        llama_table_name = f"data_{self.table_name}"
        logger.info(f"Deleting table {llama_table_name}")
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {llama_table_name}"))
            conn.commit()

    def _ensure_vector_extension(self):
        """Ensure pgvector extension is enabled"""
        from sqlalchemy import create_engine
        
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()


    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by its ID"""
        return self.document_map.get(doc_id)
    
    def index_nodes(self, documents: List[Document], nodes: List[BaseNode]) -> bool:
        page_labels = {}
        for idx, doc in enumerate(documents):
            if ('page_label' in doc.metadata):
                page_labels[idx] = doc.metadata['page_label']

        try:
            logger.info(f"Number of nodes  {len(nodes)}")
            for idx, node in enumerate(nodes):
                if (idx in page_labels):
                    node.metadata['page_label'] = page_labels[idx]

            if not self.doc_store.get_ref_doc_info(documents[0].get_doc_id()):
                storage_context = self.get_storage_context()

                # **Load existing index or create a new one**
                embedding_model = get_embedding_model()
                index_id = self.get_existing_index_id()
                if index_id:
                    index = load_index_from_storage(storage_context=storage_context,
                                                    embed_model=embedding_model,
                                                    index_id=index_id, 
                                                    store_nodes_override=True)
                    logger.info(f"Loaded existing index with ID: {index_id}")
                else:
                    # Create a new index if no index exists
                    logger.info("Creating new index")
                    index = VectorStoreIndex([], 
                                             storage_context=storage_context, 
                                             embed_model=embedding_model,
                                             store_nodes_override=True)
                    logger.info("Created a new index")
                
                # Add nodes to the existing or new index
                index.insert_nodes(nodes)
                logger.info(f"Inserted {len(nodes)} nodes into index {index.index_id}")
                storage_context.persist(persist_dir=STORAGE_DIR)
                return True
        except Exception as e:
            logger.error(f"Error during document indexing or embedding process: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to index or embed document: {e}")
        
        logger.warning(f"Failed to create index.")
        return False
    
    def get_storage_context(self):
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.doc_store,
                index_store=self.index_store
            )
            return storage_context
        except Exception as e:
            logger.error(f"Error creating storage context: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to create storage context: {str(e)}")
        
    def get_existing_index_id(self) -> Optional[str]:
        try:
            # Fetch all index structures from the MongoDBIndexStore
            index_structs = self.index_store.index_structs()

            # Check if there are any indexes present
            if index_structs:
                index_id = index_structs[0].index_id  # Retrieve the first index_id 
                logger.info(f"Found existing index with ID: {index_id}")
                return index_id
            else:
                logger.warning("No existing index found in Index Store.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving existing index ID from Index Store: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve index ID from Index Store: {e}")
    
    def get_ref_doc_id(self, file_name: str) -> List[str]:
        """
        Return all ref_doc_ids where the metadata matches the given file_name.
        """
        storage_context = self.get_storage_context()
        docstore = storage_context.docstore

        all_ref_info = docstore.get_all_ref_doc_info() or {}
        matching_ids = []

        for ref_doc_id, ref_info in all_ref_info.items():
            # Access metadata dictionary correctly
            if ref_info.metadata.get("file_name") == file_name:
                matching_ids.append(ref_doc_id)

        return matching_ids



           
    def _load_stores(self):
        """Load stores from files if they exist, otherwise keep empty stores"""
        logger.info("Loading doc store and index store")
        docstore_path = os.path.join(STORAGE_DIR, "docstore.json")
        index_store_path = os.path.join(STORAGE_DIR, "index_store.json")
        
        if os.path.exists(docstore_path):
            self.doc_store = SimpleDocumentStore.from_persist_path(docstore_path)
        
        if os.path.exists(index_store_path):
            self.index_store = SimpleIndexStore.from_persist_path(index_store_path)

    def persist(self):
        pass

    def retrieve_from_index(self, query:str, top_k: int) -> List[Dict]:
        logger.info(f"Retrieving data for query: {query}")
        logger.info(f"Docstore contains {len(self.doc_store.docs)} documents")
        try:
            embedding_model = get_embedding_model()
            # Re-create the storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
                                                           docstore=self.doc_store,
                                                           index_store=self.index_store)

            # Retrieve the existing index ID from MongoDBIndexStore
            index_id = self.get_existing_index_id()
            if not index_id:
                raise ValueError("No index found in the index storage.")

            # Load the index from storage
            vector_index = load_index_from_storage(storage_context=storage_context,
                                                   embed_model=embedding_model,
                                                   index_id=index_id)

            # Create the retriever for the index
            retriever = vector_index.as_retriever(verbose=True, similarity_top_k=top_k)

            # Perform the retrieval with the user query
            retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query)
            logger.info(f"Number of retrieved nodes {len(retrieved_nodes)}")
            results = []
            for node_with_score in retrieved_nodes:
                node = node_with_score.node
                # Get document reference
                doc = None
                doc = self.doc_store.get_document(node.node_id)
               
                # Prepare metadata
                metadata = {}
                if hasattr(node, 'metadata'):   
                    metadata.update(node.metadata)
                if doc and hasattr(doc, 'metadata'):
                    metadata.update(doc.metadata or {})
                
                # Format result
                results.append({
                    'score': float(node_with_score.score),
                    'document': doc,
                    'type': 'vector',
                    'metadata': metadata,
                    'doc_id': node.node_id  # Fall back to node ID if no doc ID
                })
            return results
        except Exception as e:
            logger.error(f"Error during retrieval process: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve data from index: {e}")
        

    def delete_by_file_name(self, file_name: str) -> bool:
        logger.info(f"Deleting file from storage: {file_name}")
        try:
            storage_context = self.get_storage_context()
            embedding_model = get_embedding_model()

            # Get existing index ID and validate
            index_id = self.get_existing_index_id()
            if not index_id:
                logger.error("Index ID not found. Cannot proceed with deletion.")
                raise ValueError("No index found. Deletion cannot proceed without an existing index.")
            
            # Load the existing index
            index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=embedding_model,
                index_id=index_id,
                store_nodes_override=True
            )
            logger.info(f"Loaded existing index with ID: {index_id}")
            ref_docs_to_delete = self.get_ref_doc_id(file_name=file_name)
            if len(ref_docs_to_delete) < 1:
                logger.warning(f"No matching document reference ID found for file: {file_name}")
                return False

            # Perform deletion
            for ref_doc_id in ref_docs_to_delete:
                index.delete_ref_doc(ref_doc_id, delete_from_docstore=True, delete_from_indexstore=True)
                logger.info(f"Successfully deleted reference document ID: {ref_doc_id} for file: {file_name}")
            
            storage_context.persist(persist_dir=STORAGE_DIR)
            return True
        except ValueError as ve:
            logger.error(f"ValueError encountered during deletion: {str(ve)}", exc_info=True)
            raise RuntimeError(f"Deletion error: {str(ve)}")
        except Exception as e:
            logger.error(f"Unexpected error during deletion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to delete from storage due to an unexpected error: {str(e)}")