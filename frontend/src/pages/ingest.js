import Layout from '@/components/Layout';
import UploadForm from '@/components/UploadForm';
import { ingestDocument, uploadGroundTruth } from '@/utils/api';

export default function Ingest() {
  return (
    <Layout>
      <h1 className="text-2xl font-bold mb-6">Document Ingestion</h1>
      
      <div className="space-y-8">
        <UploadForm 
          title="Upload Document for RAG" 
          onUpload={ingestDocument} 
        />
        
        <UploadForm 
          title="Upload Ground Truth Data" 
          onUpload={uploadGroundTruth} 
        />
      </div>
      
      <div className="mt-8 bg-blue-50 p-4 rounded-lg">
        <h3 className="font-medium mb-2">Instructions</h3>
        <ul className="list-disc pl-5 space-y-1">
          <li>Documents: PDF, TXT, or Word documents</li>
          <li>Ground Truth: JSON file with question/answer pairs</li>
          <li>Files will be processed and added to the vector database</li>
        </ul>
      </div>
    </Layout>
  );
}