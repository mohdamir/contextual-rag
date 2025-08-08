import os
from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode
from app.core.llms import query_ollama
from app.core.llms import embedding_model
from llama_index.core.llms import ChatMessage, TextBlock
import copy
from dotenv import load_dotenv

load_dotenv()

class AnthropicContextAugmenter:
    """Anthropic-style contextual augmentation using Claude LLM"""
    def __init__(self):

        self.system_prompt="You are helpful AI Assitant."

        self.prompt_document = """<document>
        {WHOLE_DOCUMENT}
        </document>"""

        self.prompt_chunk = """Here is the chunk we want to situate within the whole document
        <chunk>
        {CHUNK_CONTENT}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    
    def create_contextual_nodes(self, nodes_, whole_document: str):
        """Function to create contextual nodes for a list of nodes"""
        nodes_modified = []
        for node in nodes_:
            new_node = copy.deepcopy(node)
            prompt = self.prompt_document.format(WHOLE_DOCUMENT=whole_document)
            prompt += "\n\n"
            prompt += self.prompt_chunk.format(CHUNK_CONTENT=node.text)
            
            context = query_ollama(prompt=prompt, system_prompt=self.system_prompt)
            new_node.metadata["context"] = context
            nodes_modified.append(new_node)
        return nodes_modified
    

    def augment_chunk(self, chunk_text: str, parent_text: str = "") -> Dict[str, str]:
        """Augment a single chunk with contextual information"""
        prompt = self.context_prompt.format(
            parent_text=parent_text[:2000],  # Limit context to avoid token limits
            chunk_text=chunk_text
        )
        
        response = self.client.completions.create(
            model="claude-2",
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            max_tokens_to_sample=500,
            temperature=0.3
        )
        
        return self._parse_response(response.completion)
    
    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """Parse the structured LLM response"""
        result = {
            "summary": "",
            "keywords": [],
            "relation": ""
        }
        
        try:
            summary_start = response_text.find("<summary>") + len("<summary>")
            summary_end = response_text.find("</summary>")
            result["summary"] = response_text[summary_start:summary_end].strip()
            
            keywords_start = response_text.find("<keywords>") + len("<keywords>")
            keywords_end = response_text.find("</keywords>")
            keywords_section = response_text[keywords_start:keywords_end].strip()
            result["keywords"] = [k.strip("- ").strip() for k in keywords_section.split("\n") if k.strip()]
            
            relation_start = response_text.find("<relation>") + len("<relation>")
            relation_end = response_text.find("</relation>")
            result["relation"] = response_text[relation_start:relation_end].strip()
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
        
        return result