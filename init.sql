-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for vector storage to test pgvector plugin
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    doc_id TEXT UNIQUE,
    embedding vector(1024),
    content TEXT,
    metadata JSONB
);

-- Create index for efficient similarity search
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
    ON document_embeddings 
    USING ivfflat (embedding vector_l2_ops) 
    WITH (lists = 100);