from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

class Pipeline:
    
    class Valves(BaseModel):
        # You can add configurable parameters here (chunk_size, model, etc.)
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_docs = {}  # Store uploaded document content keyed by user id

    async def on_startup(self):
        print("[CustomRAGPipeline] Starting up...")

    async def on_shutdown(self):
        print("[CustomRAGPipeline] Shutting down...")

    async def inlet(self, body: dict, user: dict) -> dict:
        """
        Called on every API request, including document uploads.
        Here, intercept uploaded documents and store for retrieval.
        """
        files = body.get("files", [])
        if files:
            user_id = user.get("id", "anonymous")
            self.user_docs.setdefault(user_id, [])
            for file in files:
                filename = file.get("filename")
                # 'data' usually holds file content base64 encoded or raw depending on OpenWebUI version
                file_data = file.get("data", {}).get("content")
                if file_data:
                    # Store minimal info: (filename, content)
                    self.user_docs[user_id].append({
                        "filename": filename,
                        "content": file_data
                    })
                    print(f"[CustomRAGPipeline] Stored document '{filename}' for user {user_id}")
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main chat handler.
        Retrieve stored documents per user and build response with context.
        """

        user_id = body.get("user", {}).get("id", "anonymous")
        docs = self.user_docs.get(user_id, [])

        # Dummy retrieval: just collect filenames as retrieved chunks
        retrieved_chunks = [doc["filename"] for doc in docs]

        retrieved_text = " | ".join(retrieved_chunks) if retrieved_chunks else "(No documents ingested)"

        # Construct a prompt-like response embedding retrieved context
        response = (
            f"Retrieved documents/context: {retrieved_text}\n\n"
            f"Your question: {user_message}\n\n"
            f"Response: This is a placeholder response combining your input "
            f"with the retrieved document names."
        )

        return response
