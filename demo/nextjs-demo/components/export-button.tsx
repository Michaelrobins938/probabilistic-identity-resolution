'use client'

import { useState } from 'react'
import { Download, FileText, BarChart3, Users } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ExportOption {
  id: string
  label: string
  description: string
  icon: typeof FileText
}

const exportOptions: ExportOption[] = [
  {
    id: 'pdf',
    label: 'PDF Report',
    description: 'Full analysis with charts and insights',
    icon: FileText,
  },
  {
    id: 'csv',
    label: 'CSV Data',
    description: 'Raw session and attribution data',
    icon: BarChart3,
  },
  {
    id: 'json',
    label: 'JSON Export',
    description: 'Complete data for API integration',
    icon: Users,
  },
]

export function ExportButton() {
  const [isOpen, setIsOpen] = useState(false)
  const [exporting, setExporting] = useState<string | null>(null)

  const handleExport = async (optionId: string) => {
    setExporting(optionId)
    
    // Simulate export delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Create a mock download
    const element = document.createElement('a')
    const fileContent = `Identity Resolution Report\nGenerated: ${new Date().toLocaleString()}\n\nThis is a demo export.`
    const file = new Blob([fileContent], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = `identity-resolution-report-${optionId}-${Date.now()}.txt`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    
    setExporting(null)
    setIsOpen(false)
  }

  return (
    <div className="relative">
      {/* Main Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
          "bg-netflix-red hover:bg-red-700 text-white",
          isOpen && "bg-red-700"
        )}
      >
        <Download size={18} />
        Export Report
      </button>

      {/* Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Dropdown Menu */}
          <div className="absolute right-0 top-full mt-2 w-72 bg-netflix-dark rounded-lg shadow-xl border border-netflix-gray/20 z-50 overflow-hidden">
            <div className="p-3 border-b border-netflix-gray/20">
              <p className="text-white font-semibold text-sm">Export Options</p>
              <p className="text-netflix-gray text-xs">Choose your export format</p>
            </div>
            
            <div className="p-2 space-y-1">
              {exportOptions.map((option) => {
                const Icon = option.icon
                const isExporting = exporting === option.id
                
                return (
                  <button
                    key={option.id}
                    onClick={() => handleExport(option.id)}
                    disabled={!!exporting}
                    className={cn(
                      "w-full flex items-start gap-3 p-3 rounded-lg transition-all",
                      "hover:bg-netflix-gray/20 text-left",
                      isExporting && "opacity-50 cursor-wait"
                    )}
                  >
                    <div className="p-2 bg-netflix-black rounded-lg">
                      <Icon size={18} className="text-netflix-red" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="text-white font-medium text-sm">
                          {option.label}
                        </span>
                        {isExporting && (
                          <div className="w-4 h-4 border-2 border-netflix-red border-t-transparent rounded-full animate-spin" />
                        )}
                      </div>
                      <p className="text-netflix-gray text-xs mt-0.5">
                        {option.description}
                      </p>
                    </div>
                  </button>
                )
              })}
            </div>
            
            <div className="p-3 border-t border-netflix-gray/20 bg-netflix-black/50">
              <p className="text-netflix-gray text-xs text-center">
                All exports include the latest data
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
