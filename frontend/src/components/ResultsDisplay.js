export default function ResultsDisplay({ results }) {
  if (!results) return null;
  
  return (
    <div className="mt-8 space-y-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Answer</h3>
        <p className="text-gray-700">{results.answer}</p>
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-2">Sources</h3>
        <div className="space-y-4">
          {results.sources.map((source, index) => (
            <div key={index} className="bg-white p-4 rounded-lg shadow">
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-medium">Source {index + 1}</p>
                  <p className="text-sm text-gray-500 mb-2">
                    Page: {source.metadata?.page || 'N/A'} | 
                    Score: {source.score?.toFixed(4) || 'N/A'}
                  </p>
                </div>
                <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
                  {source.score?.toFixed(4) || 'N/A'}
                </span>
              </div>
              <p className="text-gray-700 mt-2">{source.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}