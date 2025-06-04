export default function EvaluationReport({ report }) {
  if (!report) return null;
  
  const metrics = [
    { name: 'Latency', value: `${report.latency.toFixed(2)}s`, color: 'bg-blue-100 text-blue-800' },
    { name: 'Similarity Score', value: report.similarity_score.toFixed(4), color: 'bg-green-100 text-green-800' },
    { name: 'Recall@K', value: report.recall_at_k.toFixed(4), color: 'bg-purple-100 text-purple-800' },
  ];
  
  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold mb-4">Evaluation Results</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((metric, index) => (
          <div key={index} className="bg-white p-4 rounded-lg shadow">
            <p className="text-sm font-medium text-gray-500">{metric.name}</p>
            <p className={`text-xl font-semibold mt-1 ${metric.color} px-3 py-1 rounded-full inline-block`}>
              {metric.value}
            </p>
          </div>
        ))}
      </div>
      
      <div className="mt-6 bg-white p-4 rounded-lg shadow">
        <h4 className="font-medium mb-2">Interpretation</h4>
        <ul className="list-disc pl-5 space-y-1">
          <li>Latency: Time to process all evaluation queries</li>
          <li>Similarity: Cosine similarity between ground truth and RAG answers (0-1 scale)</li>
          <li>Recall@K: Proportion of relevant documents retrieved in top K results</li>
        </ul>
      </div>
    </div>
  );
}