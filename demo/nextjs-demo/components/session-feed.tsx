'use client'

import { useEffect, useRef } from 'react'
import { useStore } from '@/lib/store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatCurrency } from '@/lib/utils'
import { Monitor, Smartphone, Tablet, ShoppingCart, Clock } from 'lucide-react'

const deviceIcons = {
  desktop: Monitor,
  mobile: Smartphone,
  tablet: Tablet,
}

export function SessionFeed() {
  const { sessions, persons } = useStore()
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [sessions])

  const getPerson = (personId: string) =>
    persons.find((p) => p.id === personId)

  const getDevice = (personId: string, deviceId: string) => {
    const person = getPerson(personId)
    return person?.devices.find((d) => d.id === deviceId)
  }

  return (
    <Card className="bg-netflix-black border-netflix-dark h-[500px] flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Live Session Stream
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden">
        <div
          ref={scrollRef}
          className="h-full overflow-y-auto space-y-2 pr-2"
        >
          {sessions.length === 0 ? (
            <div className="text-center py-8 text-netflix-gray">
              Start simulation to see sessions
            </div>
          ) : (
            sessions.map((session) => {
              const person = getPerson(session.personId)
              const device = getDevice(session.personId, session.deviceId)
              const DeviceIcon = device ? deviceIcons[device.type] : Monitor

              return (
                <div
                  key={session.id}
                  className="flex items-center gap-3 p-3 rounded-lg bg-netflix-dark animate-fade-in"
                >
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-xs text-white font-bold"
                    style={{ backgroundColor: person?.color || '#666' }}
                  >
                    {person?.avatar}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-white text-sm font-medium">
                        {person?.name}
                      </span>
                      <DeviceIcon size={14} className="text-netflix-gray" />
                      <span className="text-netflix-gray text-xs">
                        {device?.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-netflix-gray">
                      <span className="flex items-center gap-1">
                        <Clock size={12} />
                        {session.duration}s
                      </span>
                      <span>{session.page}</span>
                      <span>from {session.referrer}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {session.converted ? (
                      <>
                        <ShoppingCart size={16} className="text-green-500" />
                        <span className="text-green-500 text-sm font-medium">
                          {formatCurrency(session.revenue || 0)}
                        </span>
                      </>
                    ) : (
                      <span className="text-netflix-gray text-xs">View</span>
                    )}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}
