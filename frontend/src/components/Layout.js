// filepath: /workspaces/contextual-rag/frontend/src/components/Layout.js
import Link from 'next/link';

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="container py-4">
          <nav className="flex space-x-6">
            <Link href="/" passHref legacyBehavior>
              <a className="font-semibold text-primary text-lg">Contextual RAG</a>
            </Link>
            <div className="flex-1"></div>
            <Link href="/" passHref legacyBehavior>
              <a className="hover:text-primary">Query</a>
            </Link>
            <Link href="/ingest" passHref legacyBehavior>
              <a className="hover:text-primary">Ingest</a>
            </Link>
            <Link href="/evaluate" passHref legacyBehavior>
              <a className="hover:text-primary">Evaluate</a>
            </Link>
          </nav>
        </div>
      </header>
      
      <main className="flex-1 py-8">
        <div className="container">
          {children}
        </div>
      </main>
      
      <footer className="bg-white border-t py-4 mt-8">
        <div className="container text-center text-gray-500">
          RAG Evaluation System © {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}