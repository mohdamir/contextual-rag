import { useState } from 'react';

export default function EvaluationForm({ onSubmit }) {
  const [topK, setTopK] = useState(3);
  const [isRunning, setIsRunning] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsRunning(true);
    await onSubmit({ top_k: topK });
    setIsRunning(false);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Run Evaluation</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="evalTopK" className="block text-sm font-medium mb-1">
            Number of sources (K)
          </label>
          <input
            type="number"
            id="evalTopK"
            min="1"
            max="10"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-20 px-3 py-2 border rounded-md"
          />
        </div>
        
        <button 
          type="submit" 
          className="btn-primary"
          disabled={isRunning}
        >
          {isRunning ? 'Evaluating...' : 'Run Evaluation'}
        </button>
      </form>
    </div>
  );
}