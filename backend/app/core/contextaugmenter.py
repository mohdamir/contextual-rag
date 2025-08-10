import os
from typing import Dict
from app.core.llms import query_ollama
import copy
import requests
from dotenv import load_dotenv
load_dotenv()

class AnthropicContextAugmenter:
    """Anthropic-style contextual augmentation using Ollama + session reuse"""
    def __init__(self):
        self.system_prompt = "You are a helpful AI Assistant."
        
        self.prompt_document = """ <document>
            {WHOLE_DOCUMENT}
            </document>"""
        
        self.prompt_chunk = """Here is the chunk we want to situate within the document
            <chunk>
            {CHUNK_CONTENT}
            </chunk>
            Please give a short succinct context to situate this chunk within the overall document 
            for the purposes of improving search retrieval of the chunk. 
            Answer only with the succinct context and nothing else."""
        
        self.session_id = None
        self.api_base = str(os.getenv("API_BASE"))
        self.api_key = str(os.getenv("API_KEY"))
        self.model = str(os.getenv("ANTHROPIC_MODEL"))

    def query_openrouter(self, prompt, system_prompt, session_id=None):
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Optionally maintain the session or conversation id, if OpenRouter supports it
        if session_id:
            payload["session_id"] = session_id  # confirm if your OpenRouter endpoint supports sessions this way

        print (payload)
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Extract the content from the response
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ConnectionError(f"Open Router connection error: {e}")



    def create_contextual_nodes(self, nodes_, whole_document: str):
        """Create contextual nodes with session-aware document caching"""
        nodes_modified = []

        # First call — load the document into Ollama's session
        if self.session_id:
            print("Sending fulldocument for memorization")
            doc_prompt = self.prompt_document.format(WHOLE_DOCUMENT=whole_document)
            self.query_openrouter(prompt=doc_prompt, system_prompt=self.system_prompt, session_id=self.session_id)

        # Subsequent calls — only send the chunk prompt
        for node in nodes_:
            new_node = copy.deepcopy(node)

            if self.session_id:
                print("Sending only chunk for augmentation")
                prompt = self.prompt_chunk.format(CHUNK_CONTENT=node.text)
            else:
                print("Sending fulldocument + chunk for augmentation")
                prompt = self.prompt_document.format(WHOLE_DOCUMENT=whole_document)
                prompt += "\n\n"
                prompt += self.prompt_chunk.format(CHUNK_CONTENT=node.text)

            context = self.query_openrouter(prompt=prompt, system_prompt=self.system_prompt, session_id=self.session_id)
            new_node.metadata["context"] = context
            nodes_modified.append(new_node)

        return nodes_modified
