FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    python3.10 \
    python3-pip \
    python3.10-venv \
    nodejs \
    npm \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    swig \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install backend requirements
WORKDIR /workspace/backend
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install frontend dependencies
WORKDIR /workspace/frontend
COPY frontend/package*.json .
RUN npm install

# Create data directories
RUN mkdir -p /workspace/backend/data/documents && \
    mkdir -p /workspace/backend/data/vector_store && \
    mkdir -p /workspace/backend/data/ground_truth

# Create startup script
WORKDIR /workspace
COPY .devcontainer/start-services.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-services.sh

ENTRYPOINT ["/usr/local/bin/start-services.sh"]