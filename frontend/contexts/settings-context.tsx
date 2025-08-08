"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

interface Settings {
  llmModel: string
  contextualModel: string
  anthropicApiKey: string
  openaiApiKey: string
  backendUrl: string
}

interface SettingsContextType {
  settings: Settings
  updateSettings: (newSettings: Partial<Settings>) => void
}

const defaultSettings: Settings = {
  llmModel: "gpt-4",
  contextualModel: "text-embedding-ada-002",
  anthropicApiKey: "",
  openaiApiKey: "",
  backendUrl: "http://localhost:8000", // Updated to your API base
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined)

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(defaultSettings)

  useEffect(() => {
    const savedSettings = localStorage.getItem("documentChatSettings")
    if (savedSettings) {
      setSettings({ ...defaultSettings, ...JSON.parse(savedSettings) })
    }
  }, [])

  const updateSettings = (newSettings: Partial<Settings>) => {
    const updatedSettings = { ...settings, ...newSettings }
    setSettings(updatedSettings)
    localStorage.setItem("documentChatSettings", JSON.stringify(updatedSettings))
  }

  return <SettingsContext.Provider value={{ settings, updateSettings }}>{children}</SettingsContext.Provider>
}

export function useSettings() {
  const context = useContext(SettingsContext)
  if (context === undefined) {
    throw new Error("useSettings must be used within a SettingsProvider")
  }
  return context
}
