"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { TestTube, CheckCircle, XCircle, Loader2, AlertCircle, FileText, Clock, RefreshCw } from 'lucide-react'
import { useSettings } from "@/contexts/settings-context"
import { DocumentAPI } from "@/lib/api"

interface TestResult {
  endpoint: string
  status: "success" | "error" | "loading"
  data?: any
  error?: string
  timestamp: string
}

export function APITestPanel() {
  const [testResults, setTestResults] = useState<TestResult[]>([])
  const [isTestingAll, setIsTestingAll] = useState(false)
  const { settings } = useSettings()

  const api = new DocumentAPI(settings.backendUrl)

  const addTestResult = (result: Omit<TestResult, "timestamp">) => {
    const newResult: TestResult = {
      ...result,
      timestamp: new Date().toLocaleTimeString()
    }
    setTestResults(prev => [newResult, ...prev])
  }

  const testHealthCheck = async () => {
    addTestResult({
      endpoint: "GET /",
      status: "loading"
    })

    try {
      const isHealthy = await api.healthCheck()
      addTestResult({
        endpoint: "GET /",
        status: isHealthy ? "success" : "error",
        data: { healthy: isHealthy },
        error: isHealthy ? undefined : "Health check returned false"
      })
    } catch (error) {
      addTestResult({
        endpoint: "GET /",
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error"
      })
    }
  }

  const testListDocuments = async () => {
    addTestResult({
      endpoint: "GET /ingest/documents",
      status: "loading"
    })

    try {
      const response = await api.listDocuments()
      addTestResult({
        endpoint: "GET /ingest/documents",
        status: "success",
        data: response
      })
    } catch (error) {
      addTestResult({
        endpoint: "GET /ingest/documents",
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error"
      })
    }
  }

  const testDeleteDocument = async () => {
    const testDocId = "test-doc-id"
    addTestResult({
      endpoint: `DELETE /ingest/documents/${testDocId}`,
      status: "loading"
    })

    try {
      await api.deleteDocument(testDocId)
      addTestResult({
        endpoint: `DELETE /ingest/documents/${testDocId}`,
        status: "success",
        data: { message: "Delete request completed" }
      })
    } catch (error) {
      addTestResult({
        endpoint: `DELETE /ingest/documents/${testDocId}`,
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error"
      })
    }
  }

  const testAllEndpoints = async () => {
    setIsTestingAll(true)
    await testHealthCheck()
    await new Promise(resolve => setTimeout(resolve, 500)) // Small delay between tests
    await testListDocuments()
    await new Promise(resolve => setTimeout(resolve, 500))
    await testDeleteDocument()
    setIsTestingAll(false)
  }

  const clearResults = () => {
    setTestResults([])
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "loading":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success":
        return "bg-green-100 text-green-800"
      case "error":
        return "bg-red-100 text-red-800"
      case "loading":
        return "bg-blue-100 text-blue-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TestTube className="h-5 w-5" />
          API Testing Panel
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Test your backend API endpoints: {settings.backendUrl}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Test Buttons */}
        <div className="flex flex-wrap gap-2">
          <Button onClick={testHealthCheck} variant="outline" size="sm">
            <CheckCircle className="h-4 w-4 mr-2" />
            Test Health Check
          </Button>
          <Button onClick={testListDocuments} variant="outline" size="sm">
            <FileText className="h-4 w-4 mr-2" />
            Test List Documents
          </Button>
          <Button onClick={testDeleteDocument} variant="outline" size="sm">
            <XCircle className="h-4 w-4 mr-2" />
            Test Delete Document
          </Button>
          <Separator orientation="vertical" className="h-6" />
          <Button 
            onClick={testAllEndpoints} 
            disabled={isTestingAll}
            className="bg-primary"
          >
            {isTestingAll ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Testing All...
              </>
            ) : (
              <>
                <TestTube className="h-4 w-4 mr-2" />
                Test All Endpoints
              </>
            )}
          </Button>
          <Button onClick={clearResults} variant="ghost" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Clear Results
          </Button>
        </div>

        {/* Test Results */}
        {testResults.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium">Test Results</h3>
            <ScrollArea className="h-96 border rounded-lg p-3">
              <div className="space-y-3">
                {testResults.map((result, index) => (
                  <div key={index} className="border rounded-lg p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(result.status)}
                        <code className="text-sm font-mono">{result.endpoint}</code>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className={`text-xs ${getStatusColor(result.status)}`}>
                          {result.status}
                        </Badge>
                        <span className="text-xs text-muted-foreground">{result.timestamp}</span>
                      </div>
                    </div>

                    {result.error && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          <strong>Error:</strong> {result.error}
                        </AlertDescription>
                      </Alert>
                    )}

                    {result.data && (
                      <div className="bg-muted rounded p-2">
                        <p className="text-xs text-muted-foreground mb-1">Response Data:</p>
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(result.data, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* API Endpoint Documentation */}
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Expected API Endpoints</h3>
          <div className="text-xs space-y-1 text-muted-foreground">
            <div><code>GET {settings.backendUrl}/</code> - Health check</div>
            <div><code>GET {settings.backendUrl}/ingest/documents</code> - List documents</div>
            <div><code>DELETE {settings.backendUrl}/ingest/documents/{{id}}</code> - Delete document</div>
            <div><code>POST {settings.backendUrl}/ingest/</code> - Upload document</div>
            <div><code>POST {settings.backendUrl}/query/</code> - Query documents</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
