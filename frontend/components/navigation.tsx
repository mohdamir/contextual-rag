"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { MessageSquare, Settings, TestTube } from 'lucide-react'

export function Navigation() {
  const pathname = usePathname()

  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <MessageSquare className="h-6 w-6" />
            <span className="font-bold">Document Chat</span>
          </Link>
        </div>
        <div className="flex items-center space-x-4">
          <Button variant={pathname === "/" ? "default" : "ghost"} size="sm" asChild>
            <Link href="/">Chat</Link>
          </Button>
          <Button variant={pathname === "/test" ? "default" : "ghost"} size="sm" asChild>
            <Link href="/test">
              <TestTube className="h-4 w-4 mr-2" />
              Test API
            </Link>
          </Button>
          <Button variant={pathname === "/settings" ? "default" : "ghost"} size="sm" asChild>
            <Link href="/settings">
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Link>
          </Button>
        </div>
      </div>
    </nav>
  )
}
