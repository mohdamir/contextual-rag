from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import httpx
import base64

BACKEND_API_BASE = "http://backend:8000"

class Pipeline:

    class Valves(BaseModel):
        backend_url: str = "http://backend:8000"
        top_k: int = 3
        model_name: str = "llama3.2:3b"

    def __init__(self):
        self.valves = self.Valves()
        # No need for user_docs dict anymore; API handles state

    async def on_startup(self):
        print("[CustomRAGPipeline] Starting up...")

    async def on_shutdown(self):
        print("[CustomRAGPipeline] Shutting down...")

    async def inlet(self, body: dict, user: dict) -> dict:
        files = body.get("files", [])
        print (f"Body: {body}")
        print (f"User: {user}")
        if files:
            for file in files:
                filename = file.get("filename")
                # Either get raw content (base64) or URL to fetch
                file_data = file.get("data", {}).get("content")
                file_url = file.get("url")
                
                print(f"File uploaded: {filename}")

                # Example: Call your ingestion API to send the file for processing
                if file_data:
                    import base64
                    import httpx
                    file_bytes = base64.b64decode(file_data)
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://backend:8000/ingest/",
                            files={"file": (filename, file_bytes, "application/octet-stream")},
                            headers={"accept": "application/json"}
                        )
                        print(f"Ingestion API response: {response.status_code}")

                elif file_url:
                    # Optionally fetch file content and send to ingestion if needed
                    pass
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        On chat, proxy the question to backend /query API and return its output.
        """
        import requests

        top_k = self.valves.top_k
        payload = {
            "query": user_message,
            "top_k": top_k
        }
        try:
            res = requests.post(
                f"{self.valves.backend_url}/query/",
                json=payload,
                headers={"accept": "application/json"},
                timeout=60,
            )
            if res.status_code == 200:
                # Return directly the backend's answer (or parse/modify as you wish)
                answer = res.json().get("answer") or res.json()
                return answer
            else:
                return f"[CustomRAGPipeline] Error: backend query failed with status {res.status_code}\n{res.text}"
        except Exception as e:
            return f"[CustomRAGPipeline] Exception during backend query: {e}"

