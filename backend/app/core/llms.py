import os
import time
from typing import List, Optional
from logging import getLogger
from dotenv import load_dotenv
import httpx
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter
from langchain_community.chat_models import ChatOpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
import json
import requests

# Configure logging
logger = getLogger(__name__)
load_dotenv()

class OllamaService:
    """Handles connections to Ollama with retry logic"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        #self._ensure_ollama_ready()
    
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


# Configure models with retry logic
def get_embedding_model() -> OllamaEmbedding:
    """Initialize embedding model with retry"""
    return OllamaEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    )

def get_ollama_llm() -> Ollama:
    """Initialize LLM with retry"""
    return Ollama(
        model=os.getenv("LLM_MODEL", "llama3"),
        base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
        request_timeout=int(os.getenv("TIMEOUT", 60))
    )

def get_chatopenai_llm() -> ChatOpenAI:
        llm = ChatOpenAI(
            model= os.getenv("OPENROUTER_MODEL"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_API_BASE"),
            temperature=float(os.getenv("LLM_TEMPERATURE"))
        )
        return llm


def get_openrouter_llm() -> OpenRouter:
    llm = OpenRouter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENROUTER_API_BASE"),
        model=os.getenv("ANTHROPIC_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE"))
        )
    return llm

def query_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    session_id: Optional[str]=None,
    model: str = None,
    stream: bool = False
) -> str:
    """
    Improved Ollama query function that handles both streaming and non-streaming responses.
    
    Args:
        prompt: User's input/question
        system_prompt: Optional system message
        model: Model name (default: llama3)
        base_url: Ollama server URL
        stream: Whether to use streaming mode
        
    Returns:
        Complete generated response
    """
    base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.5))
    request_timeout=int(os.getenv("TIMEOUT", 60))
    if model is None:
        model = os.getenv("LLM_MODEL", "llama3")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature":temperature
            }
        }
        if session_id:
            payload["options"]["conversation"] = session_id
            payload["num_ctx"]= 32768
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=request_timeout
        )
        response.raise_for_status()
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        full_response += chunk["message"]["content"]
                    elif "response" in chunk:
                        full_response += chunk["response"]
            return full_response
        else:
            data = response.json()
            return data.get("message", {}).get("content", data.get("response", ""))
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama connection error: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid response format: {e}")


embedding_model = get_embedding_model()
ollama_llm = get_ollama_llm()
ollama_service = OllamaService()
