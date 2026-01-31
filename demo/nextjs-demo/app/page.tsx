'use client'

import { HouseholdSimulator } from '@/components/household-simulator'
import { SessionFeed } from '@/components/session-feed'
import { ClusteringViz } from '@/components/clustering-viz'
import { AttributionDashboard } from '@/components/attribution-dashboard'

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      {/* Header */}
      <header className="bg-gradient-to-b from-black via-black/80 to-transparent">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
            Identity Resolution Demo
          </h1>
          <p className="text-lg text-netflix-light max-w-3xl">
            Visualize probabilistic identity resolution and multi-touch attribution 
            across a simulated household with multiple users and devices.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <HouseholdSimulator />
            <SessionFeed />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <ClusteringViz />
            <AttributionDashboard />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-netflix-dark mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-netflix-gray text-sm text-center">
            Probabilistic Identity Resolution Demo — Built with Next.js 14, D3.js, Recharts, and Zustand
          </p>
        </div>
      </footer>
    </main>
  )
}
