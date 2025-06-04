FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/documents
RUN mkdir -p /app/data/vector_store
RUN mkdir -p /app/data/ground_truth

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]