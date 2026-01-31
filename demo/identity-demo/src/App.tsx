import React from 'react';
import { SessionFeed } from '@/components/SessionFeed';
import { AttributionDashboard } from '@/components/AttributionDashboard';
import { HouseholdSimulator } from '@/components/HouseholdSimulator';
import { ClusteringViz } from '@/components/ClusteringViz';
import { Card } from '@/components/ui/Card';
import { Activity, Users, BarChart3, Brain } from 'lucide-react';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Identity Resolution</h1>
                <p className="text-xs text-gray-500">Probabilistic Person-Level Attribution</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="hidden sm:flex items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-green-500" />
                  <span>Real-time</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Users className="w-4 h-4 text-blue-500" />
                  <span>Multi-person</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-purple-500" />
                  <span>Attribution</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="mb-8">
          <Card className="p-8 bg-gradient-to-r from-indigo-600 to-purple-700 text-white border-none">
            <div className="max-w-3xl">
              <h2 className="text-3xl font-bold mb-3">
                Probabilistic Identity Resolution Demo
              </h2>
              <p className="text-indigo-100 text-lg mb-6">
                See how machine learning assigns streaming sessions to individual household members 
                based on behavioral patterns, enabling precise person-level marketing attribution.
              </p>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2 bg-white/10 rounded-full px-4 py-2">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span>Live Session Simulation</span>
                </div>
                <div className="flex items-center gap-2 bg-white/10 rounded-full px-4 py-2">
                  <span>🎯 80%+ Accuracy</span>
                </div>
                <div className="flex items-center gap-2 bg-white/10 rounded-full px-4 py-2">
                  <span>📊 Real-time Attribution</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Household Members - Left Column */}
          <div className="lg:col-span-1">
            <HouseholdSimulator />
          </div>

          {/* Session Feed - Right Column (2/3 width) */}
          <div className="lg:col-span-2">
            <SessionFeed />
          </div>
        </div>

        {/* Clustering Visualization */}
        <div className="mb-8">
          <ClusteringViz />
        </div>

        {/* Attribution Dashboard */}
        <div className="mb-8">
          <AttributionDashboard />
        </div>

        {/* How It Works Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="p-6">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4">
              <span className="text-2xl">📱</span>
            </div>
            <h3 className="font-bold text-gray-900 mb-2">1. Session Collection</h3>
            <p className="text-sm text-gray-600">
              Anonymous sessions are captured with behavioral signals: device type, content genre, 
              time-of-day, and viewing duration.
            </p>
          </Card>

          <Card className="p-6">
            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-4">
              <span className="text-2xl">🧠</span>
            </div>
            <h3 className="font-bold text-gray-900 mb-2">2. ML Assignment</h3>
            <p className="text-sm text-gray-600">
              Our ML model calculates probabilities based on historical patterns, 
              device preferences, and temporal behavior to identify the person.
            </p>
          </Card>

          <Card className="p-6">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4">
              <span className="text-2xl">📈</span>
            </div>
            <h3 className="font-bold text-gray-900 mb-2">3. Attribution</h3>
            <p className="text-sm text-gray-600">
              Conversions are attributed to specific persons within a household, 
              enabling targeted marketing and accurate ROI measurement.
            </p>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-gray-500">
              Identity Resolution Demo — Probabilistic person-level attribution system
            </p>
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span>Built with React + TypeScript</span>
              <span>•</span>
              <span>Powered by Zustand</span>
              <span>•</span>
              <span>Visualized with D3 + Recharts</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
