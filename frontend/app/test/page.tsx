"use client"

import { Navigation } from "@/components/navigation"
import { APITestPanel } from "@/components/api-test-panel"

export default function TestPage() {
  return (
    <div className="flex flex-col h-screen">
      <Navigation />
      
      <div className="flex-1 container mx-auto p-4 max-w-4xl">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold">API Testing</h1>
            <p className="text-muted-foreground">
              Test your backend API endpoints to ensure proper integration
            </p>
          </div>
          
          <APITestPanel />
        </div>
      </div>
    </div>
  )
}
