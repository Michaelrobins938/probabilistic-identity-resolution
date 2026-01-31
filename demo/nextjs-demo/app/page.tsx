'use client'

import { useState } from 'react'
import { HouseholdSimulator } from '@/components/household-simulator'
import { SessionFeed } from '@/components/session-feed'
import { ClusteringViz } from '@/components/clustering-viz'
import { AttributionDashboard } from '@/components/attribution-dashboard'
import { PerformanceMetrics } from '@/components/performance-metrics'
import { ConfidenceViz } from '@/components/confidence-viz'
import { WhatIfComparison } from '@/components/what-if-comparison'
import { ExportButton } from '@/components/export-button'
import { MissionControlHeader } from '@/components/mission-control-header'
import { SocialProofBanner } from '@/components/social-proof-banner'
import { ShareButton } from '@/components/share-button'
import { SystemTerminal } from '@/components/system-terminal'
import { DataFlowPipeline } from '@/components/data-flow-pipeline'
import { InteractiveTour } from '@/components/interactive-tour'
import { InfoPanel } from '@/components/info-panel'
import { ArchitectureView } from '@/components/architecture-view'
import { HelpCircle, Info, Layers } from 'lucide-react'

export default function Home() {
  const [showTour, setShowTour] = useState(false)
  const [showInfoPanel, setShowInfoPanel] = useState(false)
  const [showArchitecture, setShowArchitecture] = useState(false)
  const [activeInfoComponent, setActiveInfoComponent] = useState<string | null>(null)

  const startTour = () => {
    setShowTour(true)
    setShowInfoPanel(false)
    setShowArchitecture(false)
  }

  const openInfoPanel = (componentId?: string) => {
    setActiveInfoComponent(componentId || null)
    setShowInfoPanel(true)
    setShowTour(false)
    setShowArchitecture(false)
  }

  const openArchitecture = () => {
    setShowArchitecture(true)
    setShowInfoPanel(false)
    setShowTour(false)
  }

  return (
    <main className="min-h-screen bg-[#0a0a0a]">
      {/* Mission Control Header */}
      <div data-tour="mission-control">
        <MissionControlHeader />
      </div>

      {/* Social Proof Banner */}
      <SocialProofBanner />

      {/* Control Bar */}
      <div className="bg-[#1a1a1a] border-b-2 border-[#404040] px-4 py-3">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-[#606060] uppercase tracking-wider">
              Demo Controls:
            </span>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={startTour}
              className="tactical-btn flex items-center gap-2"
            >
              <HelpCircle className="w-4 h-4" />
              <span>Start Tour</span>
            </button>
            
            <button
              onClick={() => openInfoPanel()}
              className="tactical-btn flex items-center gap-2"
            >
              <Info className="w-4 h-4" />
              <span>Documentation</span>
            </button>
            
            <button
              onClick={openArchitecture}
              className="tactical-btn flex items-center gap-2"
            >
              <Layers className="w-4 h-4" />
              <span>Architecture</span>
            </button>
            
            <ShareButton />
            
            <ExportButton />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Top Row: System Status & Data Pipeline */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2" data-tour="data-pipeline">
            <DataFlowPipeline />
          </div>
          <div data-tour="performance-metrics">
            <PerformanceMetrics />
          </div>
        </div>

        {/* Middle Row: Terminal & Household */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1" data-tour="system-terminal">
            <SystemTerminal />
          </div>
          <div className="lg:col-span-2" data-tour="household">
            <HouseholdSimulator />
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <div data-tour="session-feed">
              <SessionFeed />
            </div>
            <div data-tour="confidence-viz">
              <ConfidenceViz />
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <div data-tour="clustering">
              <ClusteringViz />
            </div>
            <div data-tour="attribution">
              <AttributionDashboard />
            </div>
          </div>
        </div>

        {/* What-If Comparison - Full Width */}
        <div data-tour="what-if">
          <WhatIfComparison />
        </div>

        {/* Footer */}
        <footer className="border-t-2 border-[#404040] mt-12 pt-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#00ff41] animate-pulse" />
                <span className="text-xs font-mono text-[#00ff41] uppercase tracking-wider">
                  System Operational
                </span>
              </div>
              <span className="text-xs font-mono text-[#606060]">
                | Build: v2.4.1-TACTICAL-EDU
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              <p className="text-xs font-mono text-[#606060] text-center md:text-right">
                PROBABILISTIC IDENTITY RESOLUTION ENGINE
              </p>
            </div>
          </div>
        </footer>
      </div>

      {/* Educational Overlays */}
      {showTour && (
        <InteractiveTour 
          onClose={() => setShowTour(false)} 
        />
      )}

      {showInfoPanel && (
        <InfoPanel 
          onClose={() => setShowInfoPanel(false)}
          initialComponent={activeInfoComponent}
        />
      )}

      {showArchitecture && (
        <ArchitectureView 
          onClose={() => setShowArchitecture(false)} 
        />
      )}
    </main>
  )
}
