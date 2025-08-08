#!/bin/bash

# Start Ollama server in background
ollama serve &

# Save server PID to wait on later
OLLAMA_PID=$!

# Function to check Ollama server readiness (poll /api/tags)
check_ready() {
  until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama server to be ready..."
    sleep 2
  done
}

# Wait until server is ready before pulling models
check_ready

# Pull models asynchronously so startup is not blocked
(
  until ollama pull llama3; do
    echo "Downloading llama3..."
    sleep 5
  done
  until ollama pull nomic-embed-text; do
    echo "Downloading nomic-embed-text..."
    sleep 5
  done
  until ollama pull llama3.2:3b; do
    echo "Downloading llama3.2:3b..."
    sleep 5
  done
) &

# Wait forever for ollama serve to exit (keeps container running)
wait $OLLAMA_PID
