from app.core.vectordb import VectorDB, IRSearchEngine
from typing import List, Dict
from llama_index.core import Document

class ContextAugmenter:
    def __init__(
        self
    ):
        self.context_prompt = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

    
    def augment_context(self, documents: List[Document], embedding_model=None) -> List[Document]:
        """
        Augment the context of each document by adding a context prompt.
        """
        if embedding_model is None:
            embedding_model = self._default_embedding_model()
        
        augmented_documents = []
        for doc in documents:
            # Generate context for the document
            doc_content = doc.text
            context = self.context_prompt.format(doc_content=doc_content)
            
            # Create a new Document with the context
            augmented_doc = Document(
                text=context,
                metadata=doc.metadata
            )
            augmented_documents.append(augmented_doc)
        
        return augmented_documents