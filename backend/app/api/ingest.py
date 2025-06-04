from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from llama_index.core import SimpleDirectoryReader
from app.core.utils import save_uploaded_file
from app.core.vectordb import FaissVectorDB, BM25TFIDFEngine
from app.core.hybridretriever import HybridRetrievalSystem
import os

router = APIRouter()
DOCUMENTS_DIR = "./data/documents"

def get_hybrid_retriever() -> HybridRetrievalSystem:
    vector_db = FaissVectorDB(dimension=1024)
    ir_engine = BM25TFIDFEngine()
    return HybridRetrievalSystem(
        vector_db=vector_db,
        ir_engine=ir_engine
    )

@router.post("/")
async def ingest_document(file: UploadFile = File(...), retriever: HybridRetrievalSystem = Depends(get_hybrid_retriever)):
    try:
        # Save uploaded file
        file_path = save_uploaded_file(file, DOCUMENTS_DIR)
        if not file_path:
            raise HTTPException(
                status_code=400, 
                detail="Failed to save uploaded file"
            )
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Add filename to metadata
        for doc in documents:
            doc.metadata["filename"] = file.filename

        retriever.index_documents(documents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "documents": len(documents)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest document: {str(e)}"
        )