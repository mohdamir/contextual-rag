"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Upload,
  Send,
  FileText,
  Loader2,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  XCircle,
  Trash2,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useSettings } from "@/contexts/settings-context"
import { DocumentAPI } from "@/lib/api"
import { Navigation } from "@/components/navigation"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { DeleteConfirmationDialog } from "@/components/delete-confirmation-dialog"

interface Message {
  id: string
  type: "user" | "assistant"
  content: string
  sources?: Array<{
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

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const { settings } = useSettings()

  const [documents, setDocuments] = useState<DocumentStatus[]>([])
  const [documentsError, setDocumentsError] = useState<string | null>(null)
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false)

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<DocumentStatus | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  const api = new DocumentAPI(settings.backendUrl)

  useEffect(() => {
    fetchDocuments()
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    try {
      await api.ingestDocument(file)
      toast({
        title: "Document uploaded successfully",
        description: `${file.name} has been processed and is ready for querying.`,
      })
      // Refresh documents list after successful upload
      fetchDocuments()
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload document",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await api.queryDocuments(input)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: response.answer,
        sources: response.sources,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      toast({
        title: "Query failed",
        description: error instanceof Error ? error.message : "Failed to query documents",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const fetchDocuments = async () => {
    setIsLoadingDocuments(true)
    setDocumentsError(null)
    try {
      const response = await api.listDocuments()
      setDocuments(response.documents)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to fetch documents"
      setDocumentsError(errorMessage)
      toast({
        title: "Failed to load documents",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsLoadingDocuments(false)
    }
  }

  const handleDeleteDocument = async (document: DocumentStatus) => {
    setDocumentToDelete(document)
    setDeleteDialogOpen(true)
  }

  const confirmDeleteDocument = async () => {
    if (!documentToDelete) return

    setIsDeleting(true)
    try {
      await api.deleteDocument(documentToDelete.id)
      toast({
        title: "Document deleted",
        description: `${documentToDelete.filename} has been successfully deleted.`,
      })
      // Refresh documents list after successful deletion
      fetchDocuments()
      setDeleteDialogOpen(false)
      setDocumentToDelete(null)
    } catch (error) {
      toast({
        title: "Delete failed",
        description: error instanceof Error ? error.message : "Failed to delete document",
        variant: "destructive",
      })
    } finally {
      setIsDeleting(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "processing":
        return <Clock className="h-4 w-4 text-yellow-500" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800"
      case "processing":
        return "bg-yellow-100 text-yellow-800"
      case "failed":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="flex flex-col h-screen">
      <Navigation />

      <div className="flex-1 container mx-auto p-4 max-w-4xl">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-full">
          {/* Upload Section */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Document
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".pdf,.txt,.doc,.docx"
                    className="hidden"
                  />
                  <Button onClick={() => fileInputRef.current?.click()} disabled={isUploading} className="w-full">
                    {isUploading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <FileText className="h-4 w-4 mr-2" />
                        Choose File
                      </>
                    )}
                  </Button>
                  <p className="text-sm text-muted-foreground">Supported formats: PDF, TXT, DOC, DOCX</p>
                </div>
              </CardContent>
            </Card>

            {/* Documents List */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Documents
                  </span>
                  <Button variant="ghost" size="sm" onClick={fetchDocuments} disabled={isLoadingDocuments}>
                    <RefreshCw className={`h-4 w-4 ${isLoadingDocuments ? "animate-spin" : ""}`} />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {documentsError && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{documentsError}</AlertDescription>
                    </Alert>
                  )}

                  {isLoadingDocuments ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Loading documents...
                    </div>
                  ) : documents.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">No documents uploaded yet</p>
                  ) : (
                    <ScrollArea className="h-64">
                      <div className="space-y-2">
                        {documents.map((doc) => (
                          <div key={doc.id} className="border rounded-lg p-3 space-y-2">
                            <div className="flex items-start justify-between">
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate" title={doc.filename}>
                                  {doc.filename}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {new Date(doc.uploaded_at).toLocaleDateString()}
                                </p>
                              </div>
                              <div className="flex items-center gap-2">
                                {getStatusIcon(doc.status)}
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleDeleteDocument(doc)}
                                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                                  title="Delete document"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>

                            <div className="flex items-center justify-between">
                              <Badge variant="secondary" className={`text-xs ${getStatusColor(doc.status)}`}>
                                {doc.status}
                              </Badge>
                              {doc.file_size && (
                                <span className="text-xs text-muted-foreground">
                                  {(doc.file_size / 1024 / 1024).toFixed(1)} MB
                                </span>
                              )}
                            </div>

                            {doc.status === "failed" && doc.error_message && (
                              <Alert variant="destructive" className="mt-2">
                                <AlertCircle className="h-3 w-3" />
                                <AlertDescription className="text-xs">{doc.error_message}</AlertDescription>
                              </Alert>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Chat Section */}
          <div className="lg:col-span-3 flex flex-col">
            <Card className="flex-1 flex flex-col">
              <CardHeader>
                <CardTitle>Chat with Documents</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <ScrollArea className="flex-1 pr-4">
                  <div className="space-y-4">
                    {messages.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        Upload a document and start asking questions about it!
                      </div>
                    ) : (
                      messages.map((message) => (
                        <div key={message.id} className="space-y-2">
                          <div className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                            <div
                              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                                message.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                              }`}
                            >
                              <p className="whitespace-pre-wrap">{message.content}</p>
                            </div>
                          </div>

                          {message.sources && message.sources.length > 0 && (
                            <div className="ml-4 space-y-2">
                              <p className="text-sm font-medium text-muted-foreground">Sources:</p>
                              {message.sources.map((source, index) => (
                                <Card key={index} className="p-3">
                                  <div className="space-y-2">
                                    <div className="flex items-center justify-between">
                                      <Badge variant="secondary">Source {index + 1}</Badge>
                                      {source.score && (
                                        <Badge variant="outline">
                                          Score: {typeof source.score === "number" ? source.score.toFixed(3) : "N/A"}
                                        </Badge>
                                      )}
                                    </div>
                                    <p className="text-sm">{source.text}</p>
                                    {Object.keys(source.metadata).length > 0 && (
                                      <div className="text-xs text-muted-foreground">
                                        {Object.entries(source.metadata).map(([key, value]) => (
                                          <span key={key} className="mr-2">
                                            {key}: {String(value)}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </Card>
                              ))}
                            </div>
                          )}
                        </div>
                      ))
                    )}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="bg-muted rounded-lg px-4 py-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                <Separator className="my-4" />

                <form onSubmit={handleSubmit} className="flex gap-2">
                  <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    disabled={isLoading}
                    className="flex-1"
                  />
                  <Button type="submit" disabled={isLoading || !input.trim()}>
                    <Send className="h-4 w-4" />
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
        <DeleteConfirmationDialog
          open={deleteDialogOpen}
          onOpenChange={setDeleteDialogOpen}
          onConfirm={confirmDeleteDocument}
          documentName={documentToDelete?.filename || ""}
          isDeleting={isDeleting}
        />
      </div>
    </div>
  )
}
