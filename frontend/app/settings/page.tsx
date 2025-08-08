"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Save, TestTube } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useSettings } from "@/contexts/settings-context"
import { DocumentAPI } from "@/lib/api"
import { Navigation } from "@/components/navigation"

export default function SettingsPage() {
  const { settings, updateSettings } = useSettings()
  const [localSettings, setLocalSettings] = useState(settings)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const { toast } = useToast()

  const handleSave = () => {
    updateSettings(localSettings)
    toast({
      title: "Settings saved",
      description: "Your settings have been saved successfully.",
    })
  }

  const testConnection = async () => {
    setIsTestingConnection(true)
    try {
      const api = new DocumentAPI(localSettings.backendUrl)
      const isHealthy = await api.healthCheck()

      if (isHealthy) {
        toast({
          title: "Connection successful",
          description: "Successfully connected to the backend API.",
        })
      } else {
        toast({
          title: "Connection failed",
          description: "Could not connect to the backend API.",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Connection failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      })
    } finally {
      setIsTestingConnection(false)
    }
  }

  const updateLocalSetting = (key: keyof typeof localSettings, value: string) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="flex flex-col h-screen">
      <Navigation />

      <div className="flex-1 container mx-auto p-4 max-w-2xl">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold">Settings</h1>
            <p className="text-muted-foreground">Configure your API keys and model preferences</p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Backend Configuration</CardTitle>
              <CardDescription>Configure the connection to your backend API</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="backendUrl">Backend URL</Label>
                <Input
                  id="backendUrl"
                  value={localSettings.backendUrl}
                  onChange={(e) => updateLocalSetting("backendUrl", e.target.value)}
                  placeholder="http://localhost:8000"
                />
              </div>
              <Button
                onClick={testConnection}
                disabled={isTestingConnection}
                variant="outline"
                className="w-full bg-transparent"
              >
                <TestTube className="h-4 w-4 mr-2" />
                {isTestingConnection ? "Testing..." : "Test Connection"}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>Choose your preferred language and embedding models</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="llmModel">LLM Model</Label>
                <Select value={localSettings.llmModel} onValueChange={(value) => updateLocalSetting("llmModel", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select LLM model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gpt-4">GPT-4</SelectItem>
                    <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                    <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                    <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                    <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                    <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="contextualModel">Contextual/Embedding Model</Label>
                <Select
                  value={localSettings.contextualModel}
                  onValueChange={(value) => updateLocalSetting("contextualModel", value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select embedding model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="text-embedding-ada-002">text-embedding-ada-002</SelectItem>
                    <SelectItem value="text-embedding-3-small">text-embedding-3-small</SelectItem>
                    <SelectItem value="text-embedding-3-large">text-embedding-3-large</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
              <CardDescription>Enter your API keys for the language models</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="openaiApiKey">OpenAI API Key</Label>
                <Input
                  id="openaiApiKey"
                  type="password"
                  value={localSettings.openaiApiKey}
                  onChange={(e) => updateLocalSetting("openaiApiKey", e.target.value)}
                  placeholder="sk-..."
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="anthropicApiKey">Anthropic API Key</Label>
                <Input
                  id="anthropicApiKey"
                  type="password"
                  value={localSettings.anthropicApiKey}
                  onChange={(e) => updateLocalSetting("anthropicApiKey", e.target.value)}
                  placeholder="sk-ant-..."
                />
              </div>
            </CardContent>
          </Card>

          <Separator />

          <div className="flex justify-end">
            <Button onClick={handleSave} className="w-full sm:w-auto">
              <Save className="h-4 w-4 mr-2" />
              Save Settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
