'use client'

import { useState, useRef, useEffect, ReactNode } from 'react'

interface EducationalContent {
  what: string
  why: string
  how: string
  technicalDetails: string
  exampleScenario: string
  icon?: ReactNode
}

interface EducationalTooltipProps {
  children: ReactNode
  content: EducationalContent
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  width?: number
}

export function EducationalTooltip({
  children,
  content,
  position = 'top',
  delay = 500,
  width = 360,
}: EducationalTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [actualPosition, setActualPosition] = useState(position)
  const [tooltipStyle, setTooltipStyle] = useState<React.CSSProperties>({})
  const triggerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const timeoutRef = useRef<NodeJS.Timeout>()

  const calculatePosition = () => {
    if (!triggerRef.current || !tooltipRef.current) return

    const triggerRect = triggerRef.current.getBoundingClientRect()
    const tooltipRect = tooltipRef.current.getBoundingClientRect()
    const margin = 16

    let pos = position
    let style: React.CSSProperties = {}

    // Check available space and adjust position
    const spaceAbove = triggerRect.top
    const spaceBelow = window.innerHeight - triggerRect.bottom
    const spaceLeft = triggerRect.left
    const spaceRight = window.innerWidth - triggerRect.right

    // Determine best position based on available space
    if (position === 'top' && spaceAbove < tooltipRect.height + margin) {
      pos = 'bottom'
    } else if (position === 'bottom' && spaceBelow < tooltipRect.height + margin) {
      pos = 'top'
    } else if (position === 'left' && spaceLeft < tooltipRect.width + margin) {
      pos = 'right'
    } else if (position === 'right' && spaceRight < tooltipRect.width + margin) {
      pos = 'left'
    }

    // Calculate coordinates
    switch (pos) {
      case 'top':
        style = {
          bottom: `${window.innerHeight - triggerRect.top + margin}px`,
          left: `${triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2}px`,
        }
        // Keep within horizontal bounds
        if (triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2 < margin) {
          style.left = `${margin}px`
        } else if (triggerRect.right + tooltipRect.width / 2 > window.innerWidth - margin) {
          style.right = `${margin}px`
          style.left = 'auto'
        }
        break

      case 'bottom':
        style = {
          top: `${triggerRect.bottom + margin}px`,
          left: `${triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2}px`,
        }
        // Keep within horizontal bounds
        if (triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2 < margin) {
          style.left = `${margin}px`
        } else if (triggerRect.right + tooltipRect.width / 2 > window.innerWidth - margin) {
          style.right = `${margin}px`
          style.left = 'auto'
        }
        break

      case 'left':
        style = {
          right: `${window.innerWidth - triggerRect.left + margin}px`,
          top: `${triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2}px`,
        }
        break

      case 'right':
        style = {
          left: `${triggerRect.right + margin}px`,
          top: `${triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2}px`,
        }
        break
    }

    setActualPosition(pos)
    setTooltipStyle(style)
  }

  useEffect(() => {
    if (isVisible) {
      // Wait for tooltip to render then calculate position
      requestAnimationFrame(() => {
        calculatePosition()
      })
    }
  }, [isVisible])

  useEffect(() => {
    const handleScroll = () => {
      if (isVisible) calculatePosition()
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', handleScroll)

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
    }
  }, [isVisible])

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true)
    }, delay)
  }

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    setIsVisible(false)
  }

  const getArrowClasses = () => {
    const base = 'absolute w-3 h-3 bg-[#1a1a1a] border-[#404040]'
    switch (actualPosition) {
      case 'top':
        return `${base} bottom-[-6px] left-1/2 -translate-x-1/2 border-b border-r rotate-45`
      case 'bottom':
        return `${base} top-[-6px] left-1/2 -translate-x-1/2 border-t border-l rotate-45`
      case 'left':
        return `${base} right-[-6px] top-1/2 -translate-y-1/2 border-t border-r rotate-45`
      case 'right':
        return `${base} left-[-6px] top-1/2 -translate-y-1/2 border-b border-l rotate-45`
    }
  }

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="inline-block"
      >
        {children}
      </div>

      {isVisible && (
        <div
          ref={tooltipRef}
          className="fixed z-[60] tactical-card"
          style={{
            width,
            maxWidth: `calc(100vw - 32px)`,
            ...tooltipStyle,
          }}
          onMouseEnter={() => setIsVisible(true)}
          onMouseLeave={handleMouseLeave}
        >
          {/* Arrow */}
          <div className={getArrowClasses()} />

          {/* Header */}
          <div className="tactical-header border-b-2 border-[#404040]">
            <div className="flex items-center gap-2">
              {content.icon || (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00bfff" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4M12 8h.01" />
                </svg>
              )}
              <span className="text-xs font-bold tracking-wider uppercase text-[#00ff41]">
                Educational Briefing
              </span>
            </div>
            <button
              onClick={() => setIsVisible(false)}
              className="text-[#606060] hover:text-[#a0a0a0] text-lg leading-none"
            >
              ×
            </button>
          </div>

          {/* Content Sections */}
          <div className="p-4 space-y-4">
            {/* What Section */}
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#00ff41]" />
                <span className="text-xs font-bold text-[#00ff41] uppercase tracking-wider">
                  What It Is
                </span>
              </div>
              <p className="text-xs text-[#e0e0e0] leading-relaxed pl-3.5">
                {content.what}
              </p>
            </div>

            {/* Why Section */}
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#ffb800]" />
                <span className="text-xs font-bold text-[#ffb800] uppercase tracking-wider">
                  Why It Matters
                </span>
              </div>
              <p className="text-xs text-[#a0a0a0] leading-relaxed pl-3.5">
                {content.why}
              </p>
            </div>

            {/* How Section */}
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#00bfff]" />
                <span className="text-xs font-bold text-[#00bfff] uppercase tracking-wider">
                  How It Works
                </span>
              </div>
              <p className="text-xs text-[#a0a0a0] leading-relaxed pl-3.5">
                {content.how}
              </p>
            </div>

            {/* Technical Details */}
            <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
              <div className="text-xs font-bold text-[#606060] uppercase tracking-wider mb-2 flex items-center gap-2">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                </svg>
                Technical Specs
              </div>
              <p className="text-xs text-[#e0e0e0]/80 font-mono leading-relaxed">
                {content.technicalDetails}
              </p>
            </div>

            {/* Example Scenario */}
            <div className="border-l-2 border-[#00ff41]/50 pl-3">
              <div className="text-xs font-bold text-[#00ff41]/70 uppercase tracking-wider mb-1">
                Real-World Example
              </div>
              <p className="text-xs text-[#a0a0a0] italic leading-relaxed">
                &ldquo;{content.exampleScenario}&rdquo;
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="tactical-header border-t-2 border-[#404040] justify-center">
            <span className="text-[10px] font-mono text-[#606060] uppercase tracking-wider">
              Press [ESC] to dismiss
            </span>
          </div>
        </div>
      )}
    </>
  )
}

// Pre-built educational content for common components
export const educationalContentLibrary = {
  clustering: {
    what: 'Machine learning algorithm that groups similar digital fingerprints into identity clusters.',
    why: 'Users often interact with multiple devices and clear cookies. Clustering maintains identity continuity across fragmented touchpoints.',
    how: 'Analyzes 47+ features including canvas fingerprinting, WebGL signatures, typing patterns, temporal behavior, and network characteristics using ensemble methods.',
    technicalDetails: 'XGBoost + Isolation Forest ensemble. Feature extraction via TensorFlow.js. Real-time clustering with DBSCAN. Latency <50ms per event.',
    exampleScenario: 'A user browses on their phone during commute, then completes purchase on laptop at home. Clustering links these sessions with 94.7% confidence despite different IPs and cleared cookies.',
  },
  attribution: {
    what: 'Multi-touch attribution system that distributes conversion credit across the customer journey.',
    why: 'Single-touch models (first/last click) miss 60%+ of influential touchpoints. Multi-touch attribution reveals the true customer journey.',
    how: 'Supports 5 models: First-touch, Last-touch, Linear, Time-decay, and Data-driven (Shapley values). Integrates with clustering for cross-device attribution.',
    technicalDetails: 'Shapley value calculation via cooperative game theory. Incrementality testing via CUPED variance reduction. Lookback window: 30-90 days configurable.',
    exampleScenario: 'A customer discovers your product via Instagram ad, researches via organic search, then converts via email link. Attribution shows each channel\'s true contribution.',
  },
  identityResolution: {
    what: 'Probabilistic matching engine that resolves anonymous visitors to known identities without relying on cookies or PII.',
    why: 'Cookie deprecation (iOS 14.5+, Chrome 2024) and privacy regulations make traditional tracking obsolete. Probabilistic methods provide a privacy-compliant alternative.',
    how: 'Combines device fingerprinting, behavioral biometrics, and graph neural networks to create persistent identifiers that survive cookie clearing and cross-context browsing.',
    technicalDetails: 'Graph neural network (GNN) with attention mechanism. Feature hashing for privacy preservation. Federated learning for model updates. Accuracy: 92-96% match rate.',
    exampleScenario: 'An anonymous visitor on mobile converts to a lead on desktop. Without identity resolution, this appears as two separate users. With it, you see the complete 2-touchpoint journey.',
  },
  dataPipeline: {
    what: 'Event-driven data pipeline that processes billions of touchpoints through real-time feature extraction and ML inference.',
    why: 'Marketing decisions require sub-second latency. Batch processing creates data gaps and missed optimization opportunities.',
    how: 'Kafka streams → Feature extraction (Flink) → Redis cache → ML inference (FastAPI) → PostgreSQL persistence. End-to-end latency: <100ms P99.',
    technicalDetails: 'Apache Kafka for streaming. Apache Flink for stateful stream processing. Redis Cluster for feature store. PostgreSQL with TimescaleDB for time-series analytics.',
    exampleScenario: 'During a flash sale, 10,000 concurrent users generate events. The pipeline processes all touchpoints in real-time, enabling instant budget reallocation to high-performing channels.',
  },
  confidenceScore: {
    what: 'Statistical confidence metric indicating the probability that a cluster assignment is correct.',
    why: 'Not all matches are equal. Confidence scores enable risk-weighted decisions and identify cases requiring manual review.',
    how: 'Calculated via ensemble agreement between multiple models, feature stability analysis, and historical validation against ground-truth deterministic matches.',
    technicalDetails: 'Monte Carlo dropout for uncertainty quantification. Calibration via Platt scaling. Thresholds: 95%+ auto-accept, 70-95% review queue, <70% flag for enrichment.',
    exampleScenario: 'A cluster with 97% confidence can be used for automated personalization. A cluster with 72% confidence may indicate a shared device and needs manual review before marketing use.',
  },
}
