'use client'

import { HouseholdSimulator } from '@/components/household-simulator'
import { SessionFeed } from '@/components/session-feed'
import { ClusteringViz } from '@/components/clustering-viz'
import { AttributionDashboard } from '@/components/attribution-dashboard'
import { PerformanceMetrics } from '@/components/performance-metrics'
import { ConfidenceViz } from '@/components/confidence-viz'
import { WhatIfComparison } from '@/components/what-if-comparison'
import { ExportButton } from '@/components/export-button'
import { MissionControlHeader } from '@/components/mission-control-header'
import { SystemTerminal } from '@/components/system-terminal'
import { DataFlowPipeline } from '@/components/data-flow-pipeline'

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0a0a0a]">
      {/* Mission Control Header */}
      <MissionControlHeader />

      {/* Main Content */}
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Top Row: System Status & Data Pipeline */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2">
            <DataFlowPipeline />
          </div>
          <div>
            <PerformanceMetrics />
          </div>
        </div>

        {/* Middle Row: Terminal & Household */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <SystemTerminal />
          </div>
          <div className="lg:col-span-2">
            <HouseholdSimulator />
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <SessionFeed />
            <ConfidenceViz />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <ClusteringViz />
            <AttributionDashboard />
          </div>
        </div>

        {/* What-If Comparison - Full Width */}
        <WhatIfComparison />

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
                | Build: v2.4.1-TACTICAL
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              <p className="text-xs font-mono text-[#606060] text-center md:text-right">
                IDENTITY RESOLUTION ENGINE — TACTICAL OPERATIONS CENTER
              </p>
              <ExportButton />
            </div>
          </div>
        </footer>
      </div>
    </main>
  )
}
