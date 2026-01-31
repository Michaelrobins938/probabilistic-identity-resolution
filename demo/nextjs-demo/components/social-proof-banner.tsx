'use client'

import { BarChart3, Shield, Zap } from 'lucide-react'

interface StatItem {
  icon: typeof BarChart3
  label: string
  value: string
}

const stats: StatItem[] = [
  { icon: BarChart3, label: 'Built for Netflix Ads', value: 'AdTech Ready' },
  { icon: Zap, label: 'Accuracy', value: '81.4%' },
  { icon: Zap, label: 'Throughput', value: '12M Events/Hour' },
  { icon: Shield, label: 'Compliance', value: 'GDPR Compliant' },
]

export function SocialProofBanner() {
  return (
    <div className="bg-[#1a1a1a] border-y border-[#404040] py-3 px-4">
      <div className="max-w-[1600px] mx-auto">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-[#606060] uppercase tracking-wider">
              System Capabilities:
            </span>
          </div>
          
          <div className="flex items-center gap-6 md:gap-8 flex-wrap">
            {stats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <div key={index} className="flex items-center gap-2">
                  <Icon size={16} className="text-[#00ff41]" />
                  <span className="text-xs font-mono text-[#a0a0a0]">
                    {stat.label === 'Built for Netflix Ads' ? (
                      <>
                        <span className="text-[#00ff41]">📊</span> Built for <span className="text-[#ffb800] font-bold">Netflix Ads</span>
                      </>
                    ) : stat.label === 'Accuracy' ? (
                      <>
                        <span className="text-[#00ff41] font-bold">81.4%</span> Accuracy
                      </>
                    ) : stat.label === 'Throughput' ? (
                      <>
                        <span className="text-[#00ff41] font-bold">12M</span> Events/Hour
                      </>
                    ) : (
                      <>
                        <span className="text-[#00ff41]">✓</span> GDPR Compliant
                      </>
                    )}
                  </span>
                </div>
              )
            })}
          </div>
          
          <div className="hidden md:flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#ffb800] animate-pulse" />
            <span className="text-xs font-mono text-[#ffb800] uppercase tracking-wider">
              Production Grade
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
