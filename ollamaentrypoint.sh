#!/bin/bash
# Start Ollama in background
ollama serve &

# Pull models
until ollama pull llama3; do
  echo "Waiting for Ollama to be ready..."
  sleep 2
done
ollama pull nomic-embed-text

# Keep container running
wait