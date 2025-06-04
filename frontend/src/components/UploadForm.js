import { useState } from 'react';

export default function UploadForm({ title, onUpload }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    
    setIsUploading(true);
    try {
      const res = await onUpload(file);
      setResult(res);
    } catch (error) {
      setResult({ status: 'error', message: 'Upload failed' });
    }
    setIsUploading(false);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Select file
          </label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full"
            required
          />
        </div>
        
        <button 
          type="submit" 
          className="btn-primary"
          disabled={isUploading}
        >
          {isUploading ? 'Uploading...' : 'Upload'}
        </button>
      </form>
      
      {result && (
        <div className={`mt-4 p-4 rounded-md ${
          result.status === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
        }`}>
          {result.message || `Uploaded ${result.items} items`}
        </div>
      )}
    </div>
  );
}