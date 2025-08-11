
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RagService



def get_rag_service():
    return RagService()

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest, rag_service: RagService = Depends(get_rag_service)):

    try:
        return rag_service.chat(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to answer query {e}"
        )



    

    