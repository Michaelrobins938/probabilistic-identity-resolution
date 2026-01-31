'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

interface TourStep {
  id: string
  title: string
  description: string
  targetElement: string
  details: string
  highlightArea?: { top?: number; left?: number; right?: number; bottom?: number; width?: number; height?: number }
}

const tourSteps: TourStep[] = [
  {
    id: 'header',
    title: 'Mission Control Header',
    description: 'The command center for monitoring system status and operations.',
    targetElement: '[data-tour="header"]',
    details: 'This displays real-time system health, uptime metrics, active alerts, and grid coordinates. It provides immediate situational awareness of the identity resolution system.',
  },
  {
    id: 'pipeline',
    title: 'Data Flow Pipeline',
    description: 'Visualizes how raw events transform into identity clusters.',
    targetElement: '[data-tour="pipeline"]',
    details: 'Events flow through: Ingestion → Feature Extraction → ML Clustering → Identity Assignment. Each stage has real-time throughput metrics and health indicators.',
  },
  {
    id: 'terminal',
    title: 'System Terminal',
    description: 'Live logs and commands for system operations.',
    targetElement: '[data-tour="terminal"]',
    details: 'Shows real-time processing logs, cluster formation events, attribution calculations, and system diagnostics. All actions are timestamped with millisecond precision.',
  },
  {
    id: 'household',
    title: 'Household Simulator',
    description: 'Simulate multi-person, multi-device scenarios.',
    targetElement: '[data-tour="household"]',
    details: 'Create realistic household scenarios with multiple people using various devices. Watch how the system resolves identities even when cookies are cleared or IP addresses shared.',
  },
  {
    id: 'sessions',
    title: 'Session Feed',
    description: 'Real-time stream of tracked user sessions.',
    targetElement: '[data-tour="sessions"]',
    details: 'Displays live session data including page views, referrers, device types, and conversion events. Shows how touchpoints are captured across the customer journey.',
  },
  {
    id: 'clustering',
    title: 'Clustering Visualization',
    description: 'See how the ML model groups similar entities.',
    targetElement: '[data-tour="clustering"]',
    details: 'Visual representation of probabilistic clustering. Similar sessions and devices are grouped based on behavioral fingerprints, temporal patterns, and device characteristics.',
  },
  {
    id: 'attribution',
    title: 'Attribution Dashboard',
    description: 'Multi-touch attribution with confidence scores.',
    targetElement: '[data-tour="attribution"]',
    details: 'Compare different attribution models: First-touch, Last-touch, Linear, Time-decay, and Data-driven. Each shows how credit is distributed across the customer journey.',
  },
  {
    id: 'roi',
    title: 'ROI Analysis',
    description: 'Compare identity resolution impact on marketing metrics.',
    targetElement: '[data-tour="roi"]',
    details: 'Side-by-side comparison showing how probabilistic identity resolution improves conversion tracking, reduces attribution gaps, and increases measurable ROI.',
  },
]

export function InteractiveTour() {
  const [isActive, setIsActive] = useState(false)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [highlightBox, setHighlightBox] = useState<DOMRect | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const tooltipRef = useRef<HTMLDivElement>(null)

  const currentStep = tourSteps[currentStepIndex]

  const updateHighlight = useCallback(() => {
    if (!isActive) return

    const target = document.querySelector(currentStep.targetElement)
    if (target) {
      const rect = target.getBoundingClientRect()
      setHighlightBox(rect)

      // Calculate tooltip position
      const tooltipHeight = tooltipRef.current?.offsetHeight || 200
      const tooltipWidth = tooltipRef.current?.offsetWidth || 400
      
      let y = rect.bottom + 20
      let x = rect.left + rect.width / 2 - tooltipWidth / 2

      // Keep tooltip within viewport
      if (y + tooltipHeight > window.innerHeight) {
        y = rect.top - tooltipHeight - 20
      }
      if (x < 20) x = 20
      if (x + tooltipWidth > window.innerWidth) {
        x = window.innerWidth - tooltipWidth - 20
      }

      setTooltipPosition({ x, y })

      // Scroll into view if needed
      if (rect.top < 0 || rect.bottom > window.innerHeight) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }
  }, [isActive, currentStep])

  useEffect(() => {
    if (isActive) {
      updateHighlight()
      window.addEventListener('scroll', updateHighlight, { passive: true })
      window.addEventListener('resize', updateHighlight)

      return () => {
        window.removeEventListener('scroll', updateHighlight)
        window.removeEventListener('resize', updateHighlight)
      }
    }
  }, [isActive, updateHighlight])

  useEffect(() => {
    if (isActive) {
      updateHighlight()
    }
  }, [currentStepIndex, isActive, updateHighlight])

  const handleNext = () => {
    if (currentStepIndex < tourSteps.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1)
    } else {
      setIsActive(false)
      setCurrentStepIndex(0)
    }
  }

  const handlePrevious = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1)
    }
  }

  const handleSkip = () => {
    setIsActive(false)
    setCurrentStepIndex(0)
  }

  const handleStart = () => {
    setIsActive(true)
    setCurrentStepIndex(0)
  }

  if (!isActive) {
    return (
      <button
        onClick={handleStart}
        className="fixed bottom-6 left-6 z-50 tactical-btn tactical-btn-primary flex items-center gap-2"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4M12 8h.01" />
        </svg>
        Start Interactive Tour
      </button>
    )
  }

  return (
    <>
      {/* Dark overlay with cutout */}
      <div className="fixed inset-0 z-50 pointer-events-none">
        <svg className="w-full h-full">
          <defs>
            <mask id="spotlight-mask">
              <rect width="100%" height="100%" fill="white" />
              {highlightBox && (
                <rect
                  x={highlightBox.left - 8}
                  y={highlightBox.top - 8}
                  width={highlightBox.width + 16}
                  height={highlightBox.height + 16}
                  rx="4"
                  fill="black"
                />
              )}
            </mask>
          </defs>
          <rect
            width="100%"
            height="100%"
            fill="rgba(0, 0, 0, 0.75)"
            mask="url(#spotlight-mask)"
          />
        </svg>
      </div>

      {/* Highlight border */}
      {highlightBox && (
        <div
          className="fixed z-50 pointer-events-none border-2 border-[#00ff41] rounded-sm"
          style={{
            top: highlightBox.top - 8,
            left: highlightBox.left - 8,
            width: highlightBox.width + 16,
            height: highlightBox.height + 16,
            boxShadow: '0 0 20px rgba(0, 255, 65, 0.4), inset 0 0 20px rgba(0, 255, 65, 0.1)',
          }}
        >
          {/* Corner accents */}
          <div className="absolute -top-1 -left-1 w-3 h-3 border-l-2 border-t-2 border-[#00ff41]" />
          <div className="absolute -top-1 -right-1 w-3 h-3 border-r-2 border-t-2 border-[#00ff41]" />
          <div className="absolute -bottom-1 -left-1 w-3 h-3 border-l-2 border-b-2 border-[#00ff41]" />
          <div className="absolute -bottom-1 -right-1 w-3 h-3 border-r-2 border-b-2 border-[#00ff41]" />
        </div>
      )}

      {/* Tooltip card */}
      <div
        ref={tooltipRef}
        className="fixed z-50 tactical-card w-[400px] max-w-[calc(100vw-40px)]"
        style={{
          left: tooltipPosition.x,
          top: tooltipPosition.y,
        }}
      >
        {/* Header */}
        <div className="tactical-header border-b-2 border-[#404040]">
          <div className="flex items-center gap-2">
            <span className="text-[#00ff41] font-bold text-lg">›</span>
            <span className="text-sm font-bold tracking-wider uppercase text-[#e0e0e0]">
              {currentStep.title}
            </span>
          </div>
          <div className="text-xs font-mono text-[#606060]">
            STEP {currentStepIndex + 1} OF {tourSteps.length}
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-3">
          <p className="text-sm text-[#a0a0a0] leading-relaxed">
            {currentStep.description}
          </p>
          
          <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
            <div className="text-xs font-bold text-[#00bfff] uppercase tracking-wider mb-2">
              Technical Details
            </div>
            <p className="text-xs text-[#e0e0e0] leading-relaxed font-mono">
              {currentStep.details}
            </p>
          </div>

          {/* Progress bar */}
          <div className="w-full h-1 bg-[#2d2d2d] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#00ff41] transition-all duration-300"
              style={{ width: `${((currentStepIndex + 1) / tourSteps.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="tactical-header border-t-2 border-[#404040] flex justify-between items-center">
          <div className="flex gap-2">
            <button
              onClick={handlePrevious}
              disabled={currentStepIndex === 0}
              className="tactical-btn text-xs py-1 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ‹ PREV
            </button>
            <button
              onClick={handleNext}
              className="tactical-btn tactical-btn-primary text-xs py-1 px-3"
            >
              {currentStepIndex === tourSteps.length - 1 ? 'FINISH' : 'NEXT ›'}
            </button>
          </div>
          <button
            onClick={handleSkip}
            className="text-xs font-mono text-[#606060] hover:text-[#a0a0a0] transition-colors"
          >
            [SKIP TOUR]
          </button>
        </div>
      </div>

      {/* Step indicator dots */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex gap-2">
        {tourSteps.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentStepIndex(index)}
            className={`w-2 h-2 rounded-full transition-all ${
              index === currentStepIndex
                ? 'bg-[#00ff41] w-6'
                : index < currentStepIndex
                ? 'bg-[#00ff41]/50'
                : 'bg-[#404040]'
            }`}
          />
        ))}
      </div>
    </>
  )
}
