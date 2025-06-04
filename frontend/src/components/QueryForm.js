import { useState } from 'react';

export default function QueryForm({ onSubmit }) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(3);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ query, top_k: topK });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="query" className="block text-sm font-medium mb-1">
          Enter your question
        </label>
        <textarea
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows="3"
          className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
          required
        />
      </div>
      
      <div>
        <label htmlFor="topK" className="block text-sm font-medium mb-1">
          Number of sources (K)
        </label>
        <input
          type="number"
          id="topK"
          min="1"
          max="10"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
          className="w-20 px-3 py-2 border rounded-md"
        />
      </div>
      
      <button type="submit" className="btn-primary">
        Submit Query
      </button>
    </form>
  );
}