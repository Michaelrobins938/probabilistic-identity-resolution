'use client'

import { useEffect, useState } from 'react'
import { useStore, generateSession } from '@/lib/store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Monitor, Smartphone, Tablet, Play, Pause, RotateCcw, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'

const deviceIcons = {
  desktop: Monitor,
  mobile: Smartphone,
  tablet: Tablet,
}

export function HouseholdSimulator() {
  const { persons, isRunning, speed, toggleSimulation, setSpeed, reset, sessions, startSimulation } = useStore()
  const [activeDevices, setActiveDevices] = useState<Set<string>>(new Set())
  const [showQuickDemo, setShowQuickDemo] = useState(false)

  // Auto-start after 2 seconds on page load
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isRunning && sessions.length === 0) {
        startSimulation()
      }
    }, 2000)
    return () => clearTimeout(timer)
  }, [])

  // Quick Demo mode - run for 30 seconds then pause
  const runQuickDemo = () => {
    setSpeed(4)
    startSimulation()
    setShowQuickDemo(true)
    
    setTimeout(() => {
      useStore.getState().stopSimulation()
      setShowQuickDemo(false)
    }, 30000)
  }

  useEffect(() => {
    if (!isRunning) {
      setActiveDevices(new Set())
      return
    }

    const interval = setInterval(() => {
      const session = generateSession(persons)
      useStore.getState().addSession(session)
      
      setActiveDevices(prev => {
        const next = new Set(prev)
        next.add(session.deviceId)
        setTimeout(() => {
          setActiveDevices(current => {
            const updated = new Set(current)
            updated.delete(session.deviceId)
            return updated
          })
        }, 1000)
        return next
      })
    }, 2000 / speed)

    return () => clearInterval(interval)
  }, [isRunning, speed, persons])

  const conversionRate = sessions.length > 0
    ? sessions.filter(s => s.converted).length / sessions.length
    : 0

  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Household Simulator
        </CardTitle>
        <div className="flex gap-2">
          <button
            onClick={toggleSimulation}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
              isRunning
                ? "bg-red-600 hover:bg-red-700 text-white"
                : "bg-netflix-red hover:bg-red-700 text-white animate-pulse"
            )}
          >
            {isRunning ? <Pause size={18} /> : <Play size={18} />}
            {isRunning ? 'Stop' : 'Start'}
          </button>
          <button
            onClick={runQuickDemo}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
              "bg-[#ffb800] hover:bg-[#e6a600] text-black",
              showQuickDemo && "ring-2 ring-[#ffb800] ring-offset-2 ring-offset-black"
            )}
          >
            <Zap size={18} />
            Run 30-Second Demo
          </button>
          <button
            onClick={reset}
            className="p-2 rounded-lg bg-netflix-dark hover:bg-gray-700 text-white transition-colors"
          >
            <RotateCcw size={18} />
          </button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-6 flex items-center justify-between">
          <div className="flex gap-4">
            <span className="text-netflix-light">Speed:</span>
            {[0.5, 1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={cn(
                  "px-2 py-1 rounded text-sm transition-colors",
                  speed === s
                    ? "bg-netflix-red text-white"
                    : "text-netflix-light hover:text-white"
                )}
              >
                {s}x
              </button>
            ))}
          </div>
          <div className="text-right">
            <span className="text-netflix-light">Conversion Rate: </span>
            <span className="text-white font-medium">
              {(conversionRate * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {persons.map((person) => (
            <div
              key={person.id}
              className="rounded-lg bg-netflix-dark p-4"
            >
              <div className="flex items-center gap-3 mb-4">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                  style={{ backgroundColor: person.color }}
                >
                  {person.avatar}
                </div>
                <div>
                  <h3 className="text-white font-medium">{person.name}</h3>
                  <p className="text-netflix-gray text-sm">
                    {person.devices.length} devices
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                {person.devices.map((device) => {
                  const Icon = deviceIcons[device.type]
                  const isActive = activeDevices.has(device.id)

                  return (
                    <div
                      key={device.id}
                      className={cn(
                        "flex items-center gap-3 p-2 rounded-lg transition-all",
                        isActive
                          ? "bg-netflix-red/20 border border-netflix-red"
                          : "bg-black/40 border border-transparent"
                      )}
                    >
                      <Icon size={18} className="text-netflix-gray" />
                      <span className="text-sm text-netflix-light">
                        {device.name}
                      </span>
                      {isActive && (
                        <div className="ml-auto w-2 h-2 rounded-full bg-netflix-red animate-pulse" />
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
