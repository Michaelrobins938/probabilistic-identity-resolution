'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowRight, DollarSign, Target, Users } from 'lucide-react'

export function WhatIfComparison() {
  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          What-If Comparison
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Account-Level (Wasted Spend) */}
          <div className="bg-netflix-dark rounded-lg p-5 border border-netflix-dark/50 opacity-75">
            <div className="flex items-center gap-2 mb-4">
              <Users className="text-netflix-gray" size={20} />
              <h3 className="text-white font-semibold">Account-Level Attribution</h3>
            </div>
            
            <div className="space-y-4">
              <div className="bg-black/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-netflix-light text-sm">Attribution Method</span>
                  <span className="text-netflix-gray text-xs">Shared Credit</span>
                </div>
                <p className="text-netflix-gray text-sm">
                  All household members share equal credit for conversions
                </p>
              </div>
              
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="text-red-400" size={16} />
                  <span className="text-red-400 font-semibold">Wasted Ad Spend</span>
                </div>
                <p className="text-netflix-gray text-sm">
                  Budget distributed across all 3 personas without precision targeting
                </p>
              </div>
              
              <div className="text-center py-2">
                <span className="text-3xl font-bold text-netflix-gray">$0</span>
                <p className="text-netflix-gray text-xs mt-1">Targeted Revenue</p>
              </div>
            </div>
          </div>

          {/* Person-Level (Targeted) */}
          <div className="bg-netflix-dark rounded-lg p-5 border-2 border-netflix-red relative">
            <div className="absolute -top-3 left-4 bg-netflix-red text-white text-xs font-semibold px-3 py-1 rounded-full">
              RECOMMENDED
            </div>
            
            <div className="flex items-center gap-2 mb-4">
              <Target className="text-netflix-red" size={20} />
              <h3 className="text-white font-semibold">Person-Level Attribution</h3>
            </div>
            
            <div className="space-y-4">
              <div className="bg-black/40 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-netflix-light text-sm">Attribution Method</span>
                  <span className="text-green-400 text-xs font-semibold">Probabilistic Match</span>
                </div>
                <p className="text-netflix-gray text-sm">
                  87% confidence matching to specific individual (Sarah)
                </p>
              </div>
              
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="text-green-400" size={16} />
                  <span className="text-green-400 font-semibold">Precision Targeting</span>
                </div>
                <p className="text-netflix-gray text-sm">
                  Ads targeted to specific buyer with personalized messaging
                </p>
              </div>
              
              <div className="text-center py-2">
                <span className="text-3xl font-bold text-netflix-red">$170</span>
                <p className="text-netflix-light text-xs mt-1">Targeted Revenue</p>
              </div>
            </div>
          </div>
        </div>

        {/* Business Impact */}
        <div className="mt-6 bg-gradient-to-r from-netflix-red/20 to-transparent rounded-lg p-4 border-l-4 border-netflix-red">
          <div className="flex items-start gap-3">
            <ArrowRight className="text-netflix-red mt-1 flex-shrink-0" size={20} />
            <div>
              <h4 className="text-white font-semibold mb-1">Business Impact</h4>
              <p className="text-netflix-light text-sm">
                Person-level attribution enables targeted campaigns that drive higher ROI. 
                By identifying Sarah as the primary decision-maker with 87% confidence, 
                marketing spend becomes 3x more efficient compared to account-level approaches.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
