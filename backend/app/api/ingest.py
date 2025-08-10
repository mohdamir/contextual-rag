from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from llama_index.core import SimpleDirectoryReader
from app.core.utils import save_uploaded_file, delete_file
from app.core.vectordb import PGVectorDB
from app.core.bm25engine import BM25TFIDFEngine
from app.core.hybridretriever import HybridRetrievalSystem
from app.core.chunker import get_chunker_from_env, PDFChunkerBase
import os
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from fastapi.responses import JSONResponse
from phoenix.otel import register
import os
import uuid
import mimetypes
from datetime import datetime
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
            nodes = chunker.parse(documents=documents)
            for node in nodes:
                node.metadata["file_name"] = file.filename
            
            retriever.index_nodes(documents=documents, nodes=nodes)
            
            return {
                "status": "success",
                "message": "File ingested successfully",
                "file_name": file.filename,
                "documents": len(documents)
            }
        else:
            return {
                "status": "success",
                "message": "File already present. No need to ingest again.",
                "file_name": file.filename,
                "documents": 0
            }
    except Exception as e:
        delete_file(filename=file.filename, directory=DOCUMENTS_DIR)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest document: {str(e)}"
        )
    

@router.get("/documents/")
async def list_documents():
    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        return JSONResponse(content={"documents": [], "total": 0})

    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)

        if os.path.isfile(file_path):
            file_stats = os.stat(file_path)
            documents.append({
                "id": f"doc_{uuid.uuid4().hex[:8]}",
                "file_name": filename,
                "status": "completed",
                "uploaded_at": datetime.utcfromtimestamp(file_stats.st_mtime).isoformat() + "Z",
                "file_size": file_stats.st_size,
                "file_type": mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            })

    return {"documents": documents, "total": len(documents)}

# Delete document endpoint  
@router.delete("/documents/{file_name}")
async def delete_document(file_name: str, retriever: HybridRetrievalSystem = Depends(get_hybrid_retriever)):
    
    deleted_from_index = retriever.delete_by_filename(file_name=file_name)
    if deleted_from_index:
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file from directory: {str(e)}")

        return {"message": "Document deleted successfully", "file_name": file_name}
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete file from vector storage:"
        )