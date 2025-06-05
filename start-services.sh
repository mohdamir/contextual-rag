#!/bin/bash

# Start FastAPI backend
cd /workspace/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Next.js frontend
cd /workspace/frontend
npm run dev -- --port 3000

# Wait for background processes
wait
