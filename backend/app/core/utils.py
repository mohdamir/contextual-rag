import os
import shutil
from pathlib import Path
from typing import Union

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if it doesn't"""
    path = Path(path) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_uploaded_file(file, directory: str) -> str:
    """Save uploaded file to specified directory"""
    ensure_directory_exists(directory)
    file_path = Path(directory) / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_path)

def load_ground_truth_files(gt_dir: str) -> list:
    """Load all ground truth files from directory"""
    gt_dir = Path(gt_dir)
    gt_data = []
    for file_path in gt_dir.glob("*.json"):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    gt_data.extend(data)
                elif isinstance(data, dict) and "items" in data:
                    gt_data.extend(data["items"])
            except json.JSONDecodeError:
                continue
    return gt_data