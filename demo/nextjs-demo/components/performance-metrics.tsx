'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity } from 'lucide-react'

interface Metric {
  label: string
  value: string
  target: string
  status: 'PASS' | 'FAIL'
}

const metrics: Metric[] = [
  {
    label: 'P99 Latency',
    value: '104ms',
    target: '< 150ms',
    status: 'PASS',
  },
  {
    label: 'Accuracy',
    value: '81.4%',
    target: '> 80%',
    status: 'PASS',
  },
  {
    label: 'Throughput',
    value: '12M/hr',
    target: '> 10M/hr',
    status: 'PASS',
  },
  {
    label: 'Brier Score',
    value: '0.12',
    target: '< 0.15',
    status: 'PASS',
  },
]

export function PerformanceMetrics() {
  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Performance Metrics
        </CardTitle>
        <Activity className="text-netflix-red" size={20} />
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {metrics.map((metric) => (
            <div
              key={metric.label}
              className="bg-netflix-dark rounded-lg p-4 border border-netflix-dark/50 hover:border-netflix-gray/30 transition-colors"
            >
              <div className="text-netflix-gray text-xs mb-1 uppercase tracking-wide">
                {metric.label}
              </div>
              <div className="text-2xl font-bold text-white mb-1">
                {metric.value}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-netflix-light">
                  Target: {metric.target}
                </span>
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    metric.status === 'PASS'
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}
                >
                  {metric.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
