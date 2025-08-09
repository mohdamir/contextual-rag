from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from llama_index.core import SimpleDirectoryReader
from app.core.utils import save_uploaded_file, delete_file
from app.core.vectordb import PGVectorDB, BM25TFIDFEngine
from app.core.hybridretriever import HybridRetrievalSystem
from app.core.chunker import get_chunker_from_env, PDFChunkerBase
import os
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from dotenv import load_dotenv
load_dotenv()

os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)  # remove manual override
os.environ.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)         # use default (gRPC)
tracer_provider = register(
    project_name="Contextual-RAG",
    auto_instrument=True,
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

POSTGRES_URL = os.getenv('DATABASE_URL')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))

router = APIRouter()

DOCUMENTS_DIR = "./data/documents"

def get_hybrid_retriever() -> HybridRetrievalSystem:
    vector_db = PGVectorDB(POSTGRES_URL, table_name="contextual_rag", embed_dim=CHUNK_SIZE, recreate_table=False)
    ir_engine = BM25TFIDFEngine()
    return HybridRetrievalSystem(
        vector_db=vector_db,
        ir_engine=ir_engine
    )

def check_file_exists(filename: str, directory: str) -> bool:
    file_path = os.path.join(directory, filename)
    return os.path.exists(file_path)

@router.post("/")
async def ingest_document(file: UploadFile = File(...), 
    retriever: HybridRetrievalSystem = Depends(get_hybrid_retriever),
    chunker:PDFChunkerBase = Depends(get_chunker_from_env)):
    try:
        if check_file_exists(file.filename, DOCUMENTS_DIR) == False:
            file_path = save_uploaded_file(file, DOCUMENTS_DIR)
            if not file_path:
                raise HTTPException(
                    status_code=400, 
                    detail="Failed to save uploaded file"
                )
            
            documents = chunker.get_documents(file_path=file_path)
            nodes = chunker.parse(file_path)
            for node in nodes:
                node.metadata["filename"] = file.filename
            
            retriever.index_nodes(documents=documents, nodes=nodes)
            
        return {
            "status": "success",
            "filename": file.filename,
            "documents": len(documents)
        }

    except Exception as e:
        delete_file(filename=file.filename, directory=DOCUMENTS_DIR)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest document: {str(e)}"
        )
    

@router.get("/documents/")
async def list_documents():
    return {
        "documents": [
            {
                "id": "doc_123",
                "filename": "example.pdf", 
                "status": "completed",
                "uploaded_at": "2024-01-15T10:30:00Z",
                "file_size": 1024000,
                "file_type": "application/pdf"
            }
        ],
        "total": 1
    }

# Delete document endpoint  
@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    return {"message": "Document deleted successfully"}