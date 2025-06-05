
# Contextual RAG with IR and Vector Embeddings

A full-stack application for document-based Question Answering with Contextual RAG

- **Backend:** Flask API for document QA
- **Frontend:** Next.js user interface

---

## Project Structure

<pre> ├── backend/ # Flask API application 
├── frontend/ # Next.js application 
├── testdata/ # Data directories (Sample pdf and groundtruth.json) 
├── Dockerfile # Dockerizes both backend and frontend in a single image 
├── LICENSE # Project license 
└── README.md # Project documentation  </pre>


---

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) (recommended)
- Or: Python 3.10+ and Node.js 16+ (for manual/local setup)
- Embedding and LLM Models (API URL and KEY)

---

## Quick Start (with Docker)

1. **Clone the repository:**
    ```
    git clone https://github.com/mohdamir/contextual-rag
    cd contextual-rag
    ```

2. **Build the Docker image:**
    ```
    docker build -t contextual-rag-bot .
    ```

3. **Run the application:**
    ```
    docker run -p 3000:3000 -p 8000:8000 --rm contextual-rag-bot
    ```

    - **Frontend:** http://localhost:3000
    - **Backend API:** http://localhost:8000

---

## Manual Setup (Local Development)

### Backend (Flask)

1. **Setup Python environment:**
    ```
    cd backend
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run the Flask server:**
    ```
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

    - Default: http://localhost:8000

---

### Frontend (Next.js)

1. **Install dependencies:**
    ```
    cd frontend
    npm install
    ```

2. **Run the Next.js development server:**
    ```
    npm run dev
    ```

    - Default: http://localhost:3000

---

## Configuration

- **Data directories:**  
  Docker will auto-create:
    - `backend/data/documents`
    - `backend/data/vector_store`
    - `backend/data/ground_truth`

- **Environment variables:**  
  Place any required variables in `.env` files inside `backend/` and `frontend/` as needed.

---

## Development Workflow

- Update code in `backend/` or `frontend/` as needed.
- For Docker, rebuild and rerun after code changes:
    ```
    docker build -t contextual-rag-bot .
    docker run -p 3000:3000 -p 8000:8000 --rm contextual-rag-bot
    ```
- For local development, restart the relevant server after changes.

---

## API Usage
### 1. Ingest Documents
Upload a document (PDF or text) to the backend for indexing.


#### POST /api/ingest

Request Example (using curl):

```
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@/path/to/your/document.pdf"
```
Description:
Uploads and indexes a document for QA retrieval.

### 2. Query the QA Bot
Ask a question and get an answer from the indexed documents.


#### POST /api/query

Request Example:

```
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple\'s total gross margin percentage in fiscal year 2024?", "top_k": 3}'
```
Response Example:
```
json
{
  "answer": "Apple's total gross margin percentage in fiscal year 2024 was 46.2%.",
  "sources": [
    {
      "text": "...",
      "metadata": {...}
    }
  ]
}
```

### 3. Evaluate the System
Evaluate retrieval and generation performance using ground truth data.

#### POST /api/evaluate

Request Example:

```
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"top_k": 3, "retrieval_method": "hybrid"}'
```
Response Example:
```
json
{
  "latency": 0.21,
  "similarity_score": 0.89,
  "recall_at_k": 0.7,
  "details": [
    {
      "question": "...",
      "ground_truth": "...",
      "rag_answer": "...",
      "similarity": 0.93,
      "recall": 1.0,
      "latency": 0.19,
      "sources": ["..."]
    }
  ]
}
```

### 4. Upload Ground Truth Data
Upload a ground truth JSON file for evaluation.

#### POST /api/ground_truth/upload

Request Example:

```
curl -X POST http://localhost:8000/api/ground_truth/upload \
  -F "file=@/path/to/groundtruth.json"
```

Description:
Uploads a ground truth file (JSON) for use in evaluation.

## Troubleshooting

- **Port conflicts:** Ensure ports 3000 and 5000 are available.
- **Node.js errors:** Use Node.js 16 or newer.
- **Python errors:** Use Python 3.10+.
- **Module not found:** Check file paths and case sensitivity, especially in imports.

---

## References

- [Next.js Documentation](https://nextjs.org/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

---

**Happy Coding!**
