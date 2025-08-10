#!/bin/bash

# Start Ollama server in background
ollama serve &

OLLAMA_PID=$!

# Function to wait until server is ready
check_ready() {
  echo "[check_ready] starting loop..."
  until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "[check_ready] waiting..."
    sleep 2
  done
  echo "[check_ready] server ready, exiting loop."
}

check_ready

# Pull base models if not already pulled
echo "Pulling base models..."
until ollama pull llama3; do
  echo "Downloading llama3 ..."
  sleep 5
done

until ollama pull nomic-embed-text; do
  echo "Downloading nomic-embed-text..."
  sleep 5
done

# Keep the Ollama server running
wait $OLLAMA_PID
