import os
import time
from typing import List, Optional
from logging import getLogger
from dotenv import load_dotenv
import httpx
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure logging
logger = getLogger(__name__)
load_dotenv()

class OllamaService:
    """Handles connections to Ollama with retry logic"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        self._ensure_ollama_ready()
    
    def _ensure_ollama_ready(self, max_retries: int = 5, delay: float = 2.0):
        """Wait for Ollama service to become available"""
        for attempt in range(max_retries):
            try:
                response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("Ollama service is ready")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"Attempt {attempt + 1}: Ollama not ready - {str(e)}")
                time.sleep(delay)
        
        raise ConnectionError(f"Failed to connect to Ollama at {self.base_url} after {max_retries} attempts")

# Initialize connection handler
ollama_service = OllamaService()

# Configure models with retry logic
def get_embedding_model() -> OllamaEmbedding:
    """Initialize embedding model with retry"""
    return OllamaEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=ollama_service.base_url
    )

def get_llm() -> Ollama:
    """Initialize LLM with retry"""
    return Ollama(
        model=os.getenv("LLM_MODEL", "llama3"),
        base_url=ollama_service.base_url,
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
        request_timeout=60.0
    )

# Initialize models
embedding_model = get_embedding_model()
llm = get_llm()

class RobustOllamaEmbedding:
    """Enhanced embedding class with error handling"""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.client = httpx.Client(base_url=ollama_service.base_url, timeout=60.0)
    
    def get_text_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    "/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch process embeddings"""
        return [self.get_text_embedding(text) for text in texts]

# Global instance with enhanced reliability
embedding_model = RobustOllamaEmbedding()