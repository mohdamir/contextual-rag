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

# Create custom models from Modelfiles if they don't already exist
# This avoids recreating if model exists

#if ! ollama list | grep -q "llama3.2-32k"; then
#  echo "Creating model llama3.2-32k from Modelfile.llama3.2-32k..."
#  ollama create -f /Modelfile.llama3.2-32k llama3.2-32k
#fi

#if ! ollama list | grep -q "qwen3-32k"; then
#  echo "Creating model qwen3-32k from Modelfile.qwen3-1.7b-32k..."
#  ollama create -f /Modelfile.qwen3-1.7b-32k qwen3-32k
#fi

# Keep the Ollama server running
wait $OLLAMA_PID
