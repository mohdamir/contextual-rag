const API_URL = 'http://localhost:8000';

export const queryDocuments = async (query, top_k = 3) => {
  const response = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k })
  });
  return response.json();
};

export const ingestDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/ingest`, {
    method: 'POST',
    body: formData
  });
  return response.json();
};

export const uploadGroundTruth = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/groundtruth/uploadgroundtruth`, {
    method: 'POST',
    body: formData
  });
  return response.json();
};

export const runEvaluation = async (top_k = 3) => {
  const response = await fetch(`${API_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ top_k })
  });
  return response.json();
};