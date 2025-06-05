FROM python:3.10-slim

# Install system dependencies, Node.js, and npm
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        swig \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Backend setup
WORKDIR /workspace/backend
COPY backend/ ./
RUN pip install --no-cache-dir -r requirements.txt

# Frontend setup
WORKDIR /workspace/frontend
COPY frontend/ ./
RUN npm install next@latest
RUN npm install

# Data directories
WORKDIR /workspace/backend/app/data
RUN mkdir -p bm25_index_store documents faiss_vector_store ground_truth

# Startup script
WORKDIR /workspace
COPY start-services.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-services.sh

EXPOSE 8000 3000

ENTRYPOINT ["/usr/local/bin/start-services.sh"]
