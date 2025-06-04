#!/bin/bash

# Start backend
cd /workspace/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start frontend
cd /workspace/frontend
npm run dev &

# Keep container running
tail -f /dev/null