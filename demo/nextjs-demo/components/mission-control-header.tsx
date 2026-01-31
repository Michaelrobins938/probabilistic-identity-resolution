'use client'

import { useState, useEffect } from 'react'

interface Alert {
  id: string
  message: string
  type: 'info' | 'warning' | 'critical'
  timestamp: Date
}

type SystemStatus = 'OPERATIONAL' | 'WARNING' | 'CRITICAL'

export function MissionControlHeader() {
  const [status, setStatus] = useState<SystemStatus>('OPERATIONAL')
  const [uptime, setUptime] = useState(0)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [currentTime, setCurrentTime] = useState(new Date())
  const [sessionId, setSessionId] = useState('OPS-INITIALIZING')
  const [gridCoords, setGridCoords] = useState({ x: 649, y: 37 })
  const [mounted, setMounted] = useState(false)

  // Generate session ID only on client to avoid hydration mismatch
  useEffect(() => {
    setSessionId('OPS-' + Math.random().toString(36).substring(2, 8).toUpperCase())
    setMounted(true)
  }, [])

  // Update time and uptime
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
      setUptime(prev => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Simulate status changes and alerts
  useEffect(() => {
    const simulateEvents = setInterval(() => {
      const rand = Math.random()
      
      // Random status changes (rare)
      if (rand < 0.02) {
        setStatus(prev => {
          if (prev === 'OPERATIONAL') return 'WARNING'
          if (prev === 'WARNING') return Math.random() > 0.5 ? 'OPERATIONAL' : 'CRITICAL'
          return 'OPERATIONAL'
        })
      }

      // Random alerts
      if (rand < 0.1) {
        const newAlert: Alert = {
          id: Math.random().toString(36).substring(2),
          message: [
            'System scan initiated',
            'Data sync completed',
            'New cluster detected',
            'Attribution model updated',
            'Session threshold reached',
          ][Math.floor(Math.random() * 5)],
          type: ['info', 'info', 'warning', 'info'][Math.floor(Math.random() * 4)] as Alert['type'],
          timestamp: new Date(),
        }
        setAlerts(prev => [newAlert, ...prev].slice(0, 3))
        
        // Auto-remove alerts after 10 seconds
        setTimeout(() => {
          setAlerts(prev => prev.filter(a => a.id !== newAlert.id))
        }, 10000)
      }

      // Update grid coordinates
      setGridCoords({
        x: Math.floor(Math.random() * 999),
        y: Math.floor(Math.random() * 999),
      })
    }, 3000)

    return () => clearInterval(simulateEvents)
  }, [])

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const formatTime = (date: Date) => {
    return date.toISOString().split('T')[1].split('.')[0]
  }

  const getStatusClass = (status: SystemStatus) => {
    switch (status) {
      case 'OPERATIONAL': return 'status-operational'
      case 'WARNING': return 'status-warning'
      case 'CRITICAL': return 'status-critical'
      default: return 'status-operational'
    }
  }

  const getAlertClass = (type: Alert['type']) => {
    switch (type) {
      case 'info': return 'alert-box-info'
      case 'warning': return 'alert-box-warning'
      case 'critical': return 'alert-box-error'
      default: return 'alert-box-info'
    }
  }

  return (
    <div className="tactical-card w-full">
      <div className="tactical-header">
        {/* Left Section - Title */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="pulse-dot status-operational"></div>
            <span className="text-lg font-bold tracking-widest text-[#e0e0e0]">
              TACTICAL OPERATIONS CENTER
            </span>
          </div>
          <div className="text-xs text-[#606060] font-mono border-l-2 border-[#404040] pl-4">
            v2.4.1-ALPHA
          </div>
        </div>

        {/* Center Section - Session Info */}
        <div className="flex items-center gap-6 text-xs font-mono">
          <div className="flex flex-col items-center">
            <span className="text-[#606060] uppercase tracking-wider">Session ID</span>
            <span className="text-[#00bfff] font-bold tracking-widest">{sessionId}</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[#606060] uppercase tracking-wider">Grid Coordinates</span>
            <span className="text-[#e0e0e0]">
              [{gridCoords.x.toString().padStart(3, '0')}, {gridCoords.y.toString().padStart(3, '0')}]
            </span>
          </div>
        </div>

        {/* Right Section - Status & Uptime */}
        <div className="flex items-center gap-4">
          <div className="flex flex-col items-end text-xs font-mono">
            <span className="text-[#606060] uppercase tracking-wider">System Uptime</span>
            <span className="terminal-text terminal-green">{formatUptime(uptime)}</span>
          </div>
          <div className="flex flex-col items-end text-xs font-mono">
            <span className="text-[#606060] uppercase tracking-wider">Zulu Time</span>
            <span className="text-[#e0e0e0]">{formatTime(currentTime)}Z</span>
          </div>
          <div className={`status-indicator ${getStatusClass(status)}`}>
            <span className="w-2 h-2 rounded-full bg-current animate-pulse"></span>
            {status}
          </div>
        </div>
      </div>

      {/* Alert Notifications Area */}
      {alerts.length > 0 && (
        <div className="border-t border-[#404040] p-2 space-y-1">
          {alerts.map(alert => (
            <div key={alert.id} className={`alert-box ${getAlertClass(alert.type)} py-2 px-3 text-xs`}>
              <span className="text-[#606060]">[{formatTime(alert.timestamp)}]</span>{' '}
              <span className="font-bold uppercase">{alert.type}:</span>{' '}
              {alert.message}
            </div>
          ))}
        </div>
      )}

      {/* Bottom Decorative Line */}
      <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-[#00ff41]/30 to-transparent"></div>
    </div>
  )
}
