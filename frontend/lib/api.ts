interface QueryRequest {
  query: string
  top_k?: number
}

interface QueryResponse {
  answer: string
  sources: Array<{
    text: string
    metadata: Record<string, any>
    score?: any
  }>
}

interface DocumentStatus {
  id: string
  filename: string
  status: "pending" | "processing" | "completed" | "failed"
  uploaded_at: string
  error_message?: string
  file_size?: number
  file_type?: string
}

interface DocumentListResponse {
  documents: DocumentStatus[]
  total: number
}

export class DocumentAPI {
  private baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  async ingestDocument(file: File): Promise<void> {
    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(`${this.baseUrl}/ingest/`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Failed to ingest document: ${response.statusText}`)
    }
  }

  async queryDocuments(query: string, topK = 3): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        top_k: topK,
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to query documents: ${response.statusText}`)
    }

    return response.json()
  }

  async listDocuments(): Promise<DocumentListResponse> {
    const response = await fetch(`${this.baseUrl}/ingest/documents`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`)
    }

    return response.json()
  }

  async deleteDocument(documentId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/ingest/documents/${documentId}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`)
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/`)
      return response.ok
    } catch {
      return false
    }
  }
}
