'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Users } from 'lucide-react'

interface ConfidenceSegment {
  name: string
  percentage: number
  color: string
}

const confidenceData: ConfidenceSegment[] = [
  { name: 'Sarah', percentage: 87, color: '#E50914' },
  { name: 'Marcus', percentage: 8, color: '#00D4AA' },
  { name: 'Emily', percentage: 5, color: '#F59E0B' },
]

export function ConfidenceViz() {
  const total = confidenceData.reduce((sum, item) => sum + item.percentage, 0)

  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Identity Confidence
        </CardTitle>
        <Users className="text-netflix-red" size={20} />
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Stacked Bar */}
        <div className="space-y-2">
          <div className="flex h-8 rounded-lg overflow-hidden">
            {confidenceData.map((segment) => (
              <div
                key={segment.name}
                style={{
                  width: `${(segment.percentage / total) * 100}%`,
                  backgroundColor: segment.color,
                }}
                className="relative group cursor-pointer transition-all duration-300 hover:opacity-90"
              >
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-white text-sm font-semibold drop-shadow-lg">
                    {segment.percentage}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 justify-center">
          {confidenceData.map((segment) => (
            <div
              key={segment.name}
              className="flex items-center gap-2 bg-netflix-dark rounded-lg px-3 py-2"
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
              <span className="text-white text-sm font-medium">
                {segment.name}
              </span>
              <span className="text-netflix-light text-sm">
                {segment.percentage}%
              </span>
            </div>
          ))}
        </div>

        {/* Summary */}
        <div className="bg-netflix-dark rounded-lg p-4 border-l-4 border-netflix-red">
          <p className="text-netflix-light text-sm">
            <span className="text-white font-semibold">Most Likely:</span>{' '}
            Sarah (87% confidence)
          </p>
          <p className="text-netflix-gray text-xs mt-1">
            Based on behavioral patterns, device fingerprinting, and historical data
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
