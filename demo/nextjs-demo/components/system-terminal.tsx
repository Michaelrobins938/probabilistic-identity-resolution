'use client'

import { useState, useEffect, useRef } from 'react'

interface LogEntry {
  id: string
  timestamp: string
  type: 'SESSION' | 'CLUSTER' | 'ATTRIBUTION' | 'SYSTEM' | 'WARNING' | 'ERROR'
  message: string
  details: Record<string, string | number>
}

export function SystemTerminal() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const terminalRef = useRef<HTMLDivElement>(null)
  const [isAutoScroll, setIsAutoScroll] = useState(true)

  // Generate random log entries
  useEffect(() => {
    const generateLog = (): LogEntry => {
      const types: LogEntry['type'][] = ['SESSION', 'CLUSTER', 'ATTRIBUTION', 'SYSTEM', 'WARNING']
      const type = types[Math.floor(Math.random() * types.length)]
      const timestamp = new Date()
      const timeStr = timestamp.toISOString().split('T')[1].split('.')[0]

      const entries: Record<LogEntry['type'], { message: string; details: Record<string, string | number> }> = {
        SESSION: {
          message: 'SESSION_ASSIGNED',
          details: {
            person: ['Sarah', 'Michael', 'Emma', 'David', 'Lisa'][Math.floor(Math.random() * 5)],
            confidence: Math.floor(75 + Math.random() * 24) + '%',
            device: ['desktop', 'mobile', 'tablet'][Math.floor(Math.random() * 3)],
          }
        },
        CLUSTER: {
          message: 'CLUSTER_UPDATE',
          details: {
            drift_score: (0.5 + Math.random() * 0.5).toFixed(2),
            clusters_active: Math.floor(3 + Math.random() * 8),
            merge_candidates: Math.floor(Math.random() * 3),
          }
        },
        ATTRIBUTION: {
          message: 'ATTRIBUTION_CALC',
          details: {
            revenue: '$' + Math.floor(50 + Math.random() * 400),
            model: ['first_touch', 'last_touch', 'linear', 'time_decay'][Math.floor(Math.random() * 4)],
            touchpoints: Math.floor(2 + Math.random() * 6),
          }
        },
        SYSTEM: {
          message: 'SYSTEM_DIAGNOSTIC',
          details: {
            cpu_load: Math.floor(20 + Math.random() * 60) + '%',
            memory: Math.floor(40 + Math.random() * 40) + '%',
            threads: Math.floor(8 + Math.random() * 16),
          }
        },
        WARNING: {
          message: 'ANOMALY_DETECTED',
          details: {
            type: 'confidence_drop',
            severity: 'medium',
            threshold: '85%',
          }
        },
        ERROR: {
          message: 'SYSTEM_ERROR',
          details: {
            code: 'ERR_' + Math.floor(1000 + Math.random() * 8999),
            module: 'attribution_engine',
          }
        }
      }

      const entry = entries[type]

      return {
        id: Math.random().toString(36).substring(2),
        timestamp: timeStr,
        type,
        message: entry.message,
        details: entry.details,
      }
    }

    // Initial logs
    const initialLogs: LogEntry[] = []
    for (let i = 0; i < 10; i++) {
      const log = generateLog()
      log.timestamp = new Date(Date.now() - (10 - i) * 2000).toISOString().split('T')[1].split('.')[0]
      initialLogs.push(log)
    }
    setLogs(initialLogs)

    // Add new log every 1.5 seconds
    const interval = setInterval(() => {
      setLogs(prev => {
        const newLog = generateLog()
        const updated = [...prev, newLog]
        // Keep only last 100 logs
        return updated.slice(-100)
      })
    }, 1500)

    return () => clearInterval(interval)
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    if (isAutoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, isAutoScroll])

  const getTypeColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'SESSION': return 'text-[#00bfff]'
      case 'CLUSTER': return 'text-[#ffb800]'
      case 'ATTRIBUTION': return 'text-[#00ff41]'
      case 'SYSTEM': return 'text-[#a0a0a0]'
      case 'WARNING': return 'text-[#ff6b35]'
      case 'ERROR': return 'text-[#ff3333]'
      default: return 'text-[#e0e0e0]'
    }
  }

  const formatDetails = (details: Record<string, string | number>) => {
    return Object.entries(details)
      .map(([key, value]) => `${key}=${value}`)
      .join(' ')
  }

  return (
    <div className="tactical-card h-[400px] flex flex-col">
      {/* Terminal Header */}
      <div className="tactical-header py-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold tracking-widest text-[#00ff41]">SYSTEM LOG</span>
          <span className="text-[10px] text-[#606060] font-mono">/var/log/tactical_ops</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-[#606060]">LOGS:</span>
            <span className="text-[#e0e0e0]">{logs.length}</span>
          </div>
          <button 
            onClick={() => setIsAutoScroll(!isAutoScroll)}
            className="tactical-btn text-[10px] py-1 px-2"
          >
            {isAutoScroll ? 'PAUSE' : 'RESUME'}
          </button>
          <div className={`w-2 h-2 rounded-full ${isAutoScroll ? 'bg-[#00ff41] animate-pulse' : 'bg-[#ffb800]'}`}></div>
        </div>
      </div>

      {/* Terminal Content */}
      <div className="flex-1 relative overflow-hidden">
        <div 
          ref={terminalRef}
          className="absolute inset-0 overflow-y-auto p-3 font-mono text-xs space-y-1 scroll-smooth"
          style={{ 
            background: '#0a0a0a',
            backgroundImage: 'linear-gradient(rgba(0, 255, 65, 0.02) 1px, transparent 1px)',
            backgroundSize: '100% 20px'
          }}
        >
          {logs.length === 0 ? (
            <div className="text-[#606060] italic">Waiting for system events...</div>
          ) : (
            logs.map((log, index) => (
              <div 
                key={log.id} 
                className="flex gap-2 hover:bg-[#1a1a1a] px-1 -mx-1 rounded"
                style={{
                  animation: index === logs.length - 1 ? 'fadeIn 0.3s ease-in' : undefined
                }}
              >
                <span className="text-[#606060] whitespace-nowrap">[{log.timestamp}]</span>
                <span className={`${getTypeColor(log.type)} font-bold whitespace-nowrap`}>
                  {log.type}
                </span>
                <span className="text-[#e0e0e0] truncate">
                  {log.message}{' '}
                  <span className="text-[#a0a0a0]">{formatDetails(log.details)}</span>
                </span>
              </div>
            ))
          )}
          <div className="flex items-center gap-1 text-[#00ff41] mt-2">
            <span className="animate-pulse">_</span>
          </div>
        </div>

        {/* Scan Line Effect */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div 
            className="absolute left-0 right-0 h-[2px] bg-[rgba(0,255,65,0.15)]"
            style={{
              animation: 'scan 6s linear infinite'
            }}
          ></div>
        </div>
      </div>

      {/* Terminal Footer */}
      <div className="border-t border-[#404040] px-3 py-2 flex items-center justify-between text-[10px] font-mono text-[#606060]">
        <div className="flex items-center gap-4">
          <span>PID: {Math.floor(1000 + Math.random() * 9000)}</span>
          <span>THREADS: 12</span>
          <span>BUFFER: 100KB</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[#00ff41]">●</span>
          <span>LIVE FEED ACTIVE</span>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  )
}
