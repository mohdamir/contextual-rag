import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAILikeEmbedding(
    api_base=os.getenv("OPENAI_LIKE_API_BASE"),
    model_name=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENAI_LIKE_API_KEY")
)

llm = OpenAILike(
    api_base=os.getenv("OPENAI_LIKE_API_BASE"),
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_LIKE_API_KEY"),
    temperature=0.3,
)