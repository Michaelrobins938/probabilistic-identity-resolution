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

  const SCROLL_PADDING = 20
  const TOOLTIP_OFFSET = 20
  const MIN_EDGE_MARGIN = 20

  const scrollElementIntoView = useCallback((element: Element) => {
    const rect = element.getBoundingClientRect()
    const scrollX = window.scrollX || window.pageXOffset
    const scrollY = window.scrollY || window.pageYOffset
    
    // Calculate ideal scroll position with padding
    let targetTop = rect.top + scrollY - SCROLL_PADDING
    let targetLeft = rect.left + scrollX - SCROLL_PADDING
    let targetBottom = rect.bottom + scrollY + SCROLL_PADDING
    let targetRight = rect.right + scrollX + SCROLL_PADDING
    
    const viewportHeight = window.innerHeight
    const viewportWidth = window.innerWidth
    const documentHeight = document.documentElement.scrollHeight
    const documentWidth = document.documentElement.scrollWidth
    
    // Determine if scrolling is needed
    const isAboveViewport = rect.top < SCROLL_PADDING
    const isBelowViewport = rect.bottom > viewportHeight - SCROLL_PADDING
    const isLeftOfViewport = rect.left < SCROLL_PADDING
    const isRightOfViewport = rect.right > viewportWidth - SCROLL_PADDING
    
    if (isAboveViewport || isBelowViewport || isLeftOfViewport || isRightOfViewport) {
      // Calculate center position for smooth scrolling
      const elementCenterY = rect.top + scrollY + rect.height / 2
      const elementCenterX = rect.left + scrollX + rect.width / 2
      
      // Determine scroll direction and position
      let scrollToY: number | undefined
      let scrollToX: number | undefined
      
      // Vertical scroll
      if (isAboveViewport) {
        scrollToY = Math.max(0, targetTop)
      } else if (isBelowViewport) {
        // If element is taller than viewport, scroll to top with padding
        if (rect.height > viewportHeight - 2 * SCROLL_PADDING) {
          scrollToY = targetTop
        } else {
          scrollToY = targetBottom - viewportHeight
        }
      }
      
      // Horizontal scroll
      if (isLeftOfViewport) {
        scrollToX = Math.max(0, targetLeft)
      } else if (isRightOfViewport) {
        if (rect.width > viewportWidth - 2 * SCROLL_PADDING) {
          scrollToX = targetLeft
        } else {
          scrollToX = targetRight - viewportWidth
        }
      }
      
      // Apply smooth scroll
      window.scrollTo({
        left: scrollToX,
        top: scrollToY,
        behavior: 'smooth'
      })
    }
  }, [])

  const calculateTooltipPosition = useCallback((targetRect: DOMRect) => {
    const tooltipHeight = tooltipRef.current?.offsetHeight || 200
    const tooltipWidth = tooltipRef.current?.offsetWidth || 400
    const viewportHeight = window.innerHeight
    const viewportWidth = window.innerWidth
    
    // Calculate available space in each direction
    const spaceAbove = targetRect.top
    const spaceBelow = viewportHeight - targetRect.bottom
    const spaceLeft = targetRect.left
    const spaceRight = viewportWidth - targetRect.right
    
    // Calculate center position horizontally
    let x = targetRect.left + targetRect.width / 2 - tooltipWidth / 2
    
    // Calculate vertical position (prefer below, fallback to above)
    let y: number
    let position: 'below' | 'above' | 'left' | 'right' = 'below'
    
    // Determine best vertical position
    if (spaceBelow >= tooltipHeight + TOOLTIP_OFFSET) {
      // Plenty of space below
      y = targetRect.bottom + TOOLTIP_OFFSET
      position = 'below'
    } else if (spaceAbove >= tooltipHeight + TOOLTIP_OFFSET) {
      // Fall back to above
      y = targetRect.top - tooltipHeight - TOOLTIP_OFFSET
      position = 'above'
    } else if (spaceRight >= tooltipWidth + TOOLTIP_OFFSET) {
      // Try right side
      y = targetRect.top + targetRect.height / 2 - tooltipHeight / 2
      x = targetRect.right + TOOLTIP_OFFSET
      position = 'right'
    } else if (spaceLeft >= tooltipWidth + TOOLTIP_OFFSET) {
      // Try left side
      y = targetRect.top + targetRect.height / 2 - tooltipHeight / 2
      x = targetRect.left - tooltipWidth - TOOLTIP_OFFSET
      position = 'left'
    } else {
      // Default to below, even if it goes off-screen
      y = targetRect.bottom + TOOLTIP_OFFSET
      position = 'below'
    }
    
    // Ensure tooltip stays within viewport horizontally
    if (x < MIN_EDGE_MARGIN) {
      x = MIN_EDGE_MARGIN
    }
    if (x + tooltipWidth > viewportWidth - MIN_EDGE_MARGIN) {
      x = viewportWidth - tooltipWidth - MIN_EDGE_MARGIN
    }
    
    // Ensure tooltip stays within viewport vertically
    if (y < MIN_EDGE_MARGIN) {
      y = MIN_EDGE_MARGIN
    }
    if (y + tooltipHeight > viewportHeight - MIN_EDGE_MARGIN) {
      y = viewportHeight - tooltipHeight - MIN_EDGE_MARGIN
    }
    
    // If tooltip is too wide for viewport, center it
    if (tooltipWidth > viewportWidth - 2 * MIN_EDGE_MARGIN) {
      x = MIN_EDGE_MARGIN
    }
    
    return { x, y, position }
  }, [])

  const updateHighlight = useCallback(() => {
    if (!isActive) return

    const target = document.querySelector(currentStep.targetElement)
    if (target) {
      // First, scroll element into view if needed
      scrollElementIntoView(target)
      
      // Wait for scroll to complete before calculating positions
      setTimeout(() => {
        const rect = target.getBoundingClientRect()
        setHighlightBox(rect)
        
        const { x, y } = calculateTooltipPosition(rect)
        setTooltipPosition({ x, y })
      }, 100)
    }
  }, [isActive, currentStep, scrollElementIntoView, calculateTooltipPosition])

  // Debounced update for scroll and resize events
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  const debouncedUpdateHighlight = useCallback(() => {
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current)
    }
    updateTimeoutRef.current = setTimeout(() => {
      updateHighlight()
    }, 150)
  }, [updateHighlight])

  useEffect(() => {
    if (isActive) {
      updateHighlight()
      window.addEventListener('scroll', debouncedUpdateHighlight, { passive: true })
      window.addEventListener('resize', debouncedUpdateHighlight)

      return () => {
        window.removeEventListener('scroll', debouncedUpdateHighlight)
        window.removeEventListener('resize', debouncedUpdateHighlight)
        if (updateTimeoutRef.current) {
          clearTimeout(updateTimeoutRef.current)
        }
      }
    }
  }, [isActive, debouncedUpdateHighlight])

  useEffect(() => {
    if (isActive) {
      updateHighlight()
    }
  }, [currentStepIndex, isActive, updateHighlight])

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current)
      }
    }
  }, [])

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
