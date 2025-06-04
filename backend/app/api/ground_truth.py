from fastapi import APIRouter, File, UploadFile, HTTPException
from app.core.utils import save_uploaded_file
from app.models.schemas import GroundTruthFile
import json
import os

router = APIRouter()
GT_DIR = "./data/ground_truth"

@router.post("/upload")
async def upload_ground_truth(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=400, 
                detail="Only JSON files are accepted"
            )
        
        # Save file
        file_path = save_uploaded_file(file, GT_DIR)
        
        # Validate JSON structure
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Validate structure
            if isinstance(data, list):
                for item in data:
                    if "question" not in item or "answer" not in item:
                        raise ValueError("Invalid ground truth structure")
            elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                for item in data["items"]:
                    if "question" not in item or "answer" not in item:
                        raise ValueError("Invalid ground truth structure")
            else:
                raise ValueError("Invalid ground truth format")
        
        return GroundTruthFile(
            filename=file.filename,
            items=data if isinstance(data, list) else data["items"]
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON format"
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400, 
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload ground truth: {str(e)}"
        )