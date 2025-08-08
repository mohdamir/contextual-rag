import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index.core.schema import Document, TextNode, NodeWithScore, BaseNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from app.core.chunker import get_chunker_from_env, PDFChunkerBase
from app.core.llms import get_embedding_model
from app.models.schemas import RetrievedChunk
from llama_index.core import load_index_from_storage
from sqlalchemy import make_url, text
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import ensure_directory_exists
import pickle
from pathlib import Path
from app.core.logger import logger
import psycopg2

VECTOR_STORE_PATH = "./data/faiss_vector_store/index.faiss"
BMI25_STORE_PATH = "./data/bm25_index_store"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Define storage paths
STORAGE_DIR = "storage"
DOCSTORE_PATH = os.path.join(STORAGE_DIR, "docstore.json")
INDEX_STORE_PATH = os.path.join(STORAGE_DIR, "index_store.json")

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)


# ====================== Abstract Interfaces ======================

class VectorDB(ABC):
    @abstractmethod
    def index_vectors(self, ids: List[str], vectors: np.ndarray, documents: List[Document]):
        pass

    def index_nodes(self, docuemnts: List[Document], nodes: List[BaseNode]) -> bool:
        pass
    
    @abstractmethod
    def search_vectors(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        pass

    @abstractmethod
    def index_file(self, file_path:str, file_metadata: dict = None) -> bool:
        pass

    @abstractmethod
    def retrieve_from_index(self, query_data) -> List[Dict]:
        pass

class IRSearchEngine(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Document]):
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
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

    def index_vectors(self, ids: List[str], vectors: np.ndarray, documents: List[Document]):
        """Index vectors in PostgreSQL with pgvector"""
        if len(vectors) == 0 or len(documents) == 0:
            logger.warning("Empty vectors or documents provided")
            return

        # Validate inputs
        if len(ids) != len(vectors) or len(vectors) != len(documents):
            raise ValueError("IDs, vectors and documents must have same length")

        successful = 0
        nodes = []
        for idx, (id_, vector, doc) in enumerate(zip(ids, vectors, documents)):
            try:

                print(f"doc.text: {doc.text}, type: {type(doc.text)}")  # Check for None or empty string
                print(f"vector: {id_}, type: {type(vector)}")  # Ensure it's a numpy array
                print(f"metadata: {doc.metadata}, type: {type(doc.metadata)}")  # Ensure it's a numpy array
                # Validate ID
                if not id_ or id_ == 'None' or doc is None:
                    raise ValueError(f"Invalid document at index {idx}")

                # Validate vector
                if vector is None:
                    raise ValueError(f"Null vector at index {idx}")
                
                if doc.metadata is None:
                    raise ValueError(f"No metadata present at index {idx}")
                    
                vector = vector.tolist() if isinstance(vector, np.ndarray) else vector
                if len(vector) != int(self.embed_dim):
                    raise ValueError(f"Invalid vector dimensions at index {idx}")

                # Validate document
                if not doc.text or not isinstance(doc.text, str):
                    raise ValueError(f"Invalid document text at index {idx}")

                # Store document reference
                self.document_map[id_] = doc

                clean_metadata = {}
                for k, v in (doc.metadata or {}).items():
                    if isinstance(v, str) and v.lower() == 'none':
                        clean_metadata[k] = None
                    else:
                        clean_metadata[k] = v
                
                # Index with pgvector
                node = TextNode(
                    id=id_,
                    text=doc.text,  # or your document content
                    embedding=vector,
                    metadata=clean_metadata
                )
                
                nodes.append(node)
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to index document {id_}: {str(e)}")
                continue
        
        try:
            for n in nodes:
                assert isinstance(n.embedding, list), f"Embedding not list: {n.embedding}"
                assert len(n.embedding) == self.embed_dim, f"Invalid embed size: {len(n.embedding)}"
            ids = self.vector_store.add(nodes=nodes)
            logger.info(f"Successfully indexed {len(ids)}/{len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to index documents in bulk: {str(e)}")


    def search_vectors(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """Search vectors in PostgreSQL with pgvector
        
        Args:
            query_vector: Embedding vector to search with
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - score: Similarity score
            - document: Retrieved document
            - type: Result type ('vector')
            - metadata: Document metadata (if available)
            
        Raises:
            ValueError: If input is invalid or search fails
        """
        try:
            # Validate inputs
            if not hasattr(self, 'vector_store'):
                raise ValueError("Vector store not initialized")
                
            if len(query_vector) == 0:
                raise ValueError("Empty query vector provided")
                
            if top_k <= 0:
                raise ValueError("top_k must be positive")

            # Convert vector to list if needed
            query_embedding = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
            # Perform similarity search
            query_results = self.vector_store.query(
                query='',  # Empty string for pure vector search
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                vector_store_query_mode="default"
            )

            results = []
            for node_with_score in query_results.nodes_with_score:
                try:
                    doc_id = node_with_score.node.node_id
                    doc = self.document_map.get(doc_id)
                    
                    if not doc:
                        continue
                        
                    results.append({
                        'score': float(node_with_score.score),
                        'document': doc,
                        'type': 'vector',
                        'metadata': getattr(doc, 'metadata', {}),
                        'id': doc_id
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process node {doc_id}: {str(e)}")
                    continue

            logger.info(f"Found {len(results)}/{top_k} vector results")
            return results

        except Exception as e:
            logger.error(
                "Vector search failed",
                exc_info=True,
                extra={
                    'query_vector_shape': query_vector.shape if hasattr(query_vector, 'shape') else None,
                    'top_k': top_k
                }
            )
            raise ValueError(f"Vector search failed: {str(e)}") from e

    @classmethod
    def load_from_disk(cls, connection_string: str, table_name: str, embed_dim: int = 1024):
        """Load existing PGVector store"""
        return cls(
            connection_string=connection_string,
            table_name=table_name,
            embed_dim=embed_dim,
            recreate_table=False
        )

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
                storage_context.persist(persist_dir="./storage")
                return True
        except Exception as e:
            logger.error(f"Error during document indexing or embedding process: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to index or embed document: {e}")
        
        logger.warning(f"Failed to create index.")
        return False
    
    def index_file(self, file_path: str, file_metadata: dict = None) -> bool:
        logger.info(f"Parsing, embedding, and adding file to vector store: {file_path}")

        document, file_extension = self.load_document(file_path)
        
        # Set custom doc_id for all file types, and handle page_label preservation for PDFs
        file_name = os.path.basename(file_path)
        page_labels = {}
        for idx, doc in enumerate(document):
            if (file_extension == 'pdf') and ('page_label' in doc.metadata):
                page_labels[idx] = doc.metadata['page_label']

        try:
            chunk_size=512
            chunk_overlap = 20
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = parser.get_nodes_from_documents(document)
            logger.info(f"Number of nodes after parsing: {len(nodes)}")
            for idx, node in enumerate(nodes):
                if (file_extension == 'pdf') and (idx in page_labels):
                    node.metadata['page_label'] = page_labels[idx]

            if not self.doc_store.get_ref_doc_info(document[0].get_doc_id()):
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
                os.remove(file_path)
                storage_context.persist(persist_dir="./storage")
                return True
        except Exception as e:
            logger.error(f"Error during document indexing or embedding process: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to index or embed document: {e}")
        
        logger.warning(f"Failed to create index for file: {file_path} as File is already indexed.")
        return False
    
    def load_document(self, file_path: str):
        file_extension = file_path.split('.')[-1].lower()
        try:
            if file_extension in ['pdf', 'docx']:
                document = SimpleDirectoryReader(input_files=[file_path], filename_as_id=False).load_data()
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error processing file: {file_path}. Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load document for indexing: {str(e)}")
        return document, file_extension
    
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
        logger.info(f"First 10 doc IDs: {list(self.doc_store.docs.keys())}")
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
                logger.info(f"Node with node id {node.node_id}")
                # Get document reference
                doc = None
                doc = self.doc_store.get_document(node.node_id)

                print (type(doc))
                
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
    

""" class FaissVectorDB(VectorDB):
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
 """
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