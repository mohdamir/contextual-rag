import os
from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode
from app.core.llms import embedding_model
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

class AnthropicContextAugmenter:
    """Anthropic-style contextual augmentation using Claude LLM"""
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.context_prompt = """
        <document_context>
        {parent_text}
        </document_context>

        <current_chunk>
        {chunk_text}
        </current_chunk>

        Please analyze this document chunk in context and generate:
        1. A 1-2 sentence summary of the key information in this chunk
        2. The most important 3-5 keywords/phrases
        3. How this relates to the broader document context

        Return your response in this exact format:
        <summary>
        [summary text]
        </summary>
        <keywords>
        - keyword1
        - keyword2
        </keywords>
        <relation>
        [relation text]
        </relation>
        """
    
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