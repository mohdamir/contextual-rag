import { useState } from 'react';
import Layout from '@/components/Layout';
import EvaluationForm from '@/components/EvaluationForm';
import EvaluationReport from '@/components/EvaluationReport';
import { runEvaluation } from '@/utils/api';

export default function Evaluate() {
  const [report, setReport] = useState(null);
  
  const handleRunEvaluation = async ({ top_k }) => {
    const res = await runEvaluation(top_k);
    setReport(res);
  };

  return (
    <Layout>
      <h1 className="text-2xl font-bold mb-6">RAG Evaluation</h1>
      
      <EvaluationForm onSubmit={handleRunEvaluation} />
      
      {report && <EvaluationReport report={report} />}
      
      <div className="mt-8 bg-blue-50 p-4 rounded-lg">
        <h3 className="font-medium mb-2">Evaluation Metrics</h3>
        <p className="mb-2">The system will compare RAG responses against ground truth using:</p>
        <ul className="list-disc pl-5 space-y-1">
          <li><strong>Latency</strong>: Time to process all queries</li>
          <li><strong>Similarity</strong>: Semantic similarity between answers</li>
          <li><strong>Recall@K</strong>: Accuracy of retrieved documents</li>
        </ul>
      </div>
    </Layout>
  );
}