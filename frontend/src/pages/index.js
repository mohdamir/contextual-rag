import { useState } from 'react';
import Layout from '@/components/Layout';
import QueryForm from '@/components/QueryForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import { queryDocuments } from '@/utils/api';

export default function Home() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleQuerySubmit = async ({ query, top_k }) => {
    setIsLoading(true);
    try {
      const res = await queryDocuments(query, top_k);
      setResults(res);
    } catch (error) {
      setResults({ error: 'Failed to get response' });
    }
    setIsLoading(false);
  };

  return (
    <Layout>
      <h1 className="text-2xl font-bold mb-6">RAG Query Interface</h1>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <QueryForm onSubmit={handleQuerySubmit} />
      </div>
      
      {isLoading && (
        <div className="mt-6 text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
          <p className="mt-2">Processing query...</p>
        </div>
      )}
      
      {results && <ResultsDisplay results={results} />}
    </Layout>
  );
}