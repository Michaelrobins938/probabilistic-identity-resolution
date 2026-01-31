'use client'

import { useMemo, useState } from 'react'
import { useStore } from '@/lib/store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatCurrency, formatPercent } from '@/lib/utils'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from 'recharts'
import { Info } from 'lucide-react'

type ModelType = 'first-touch' | 'last-touch' | 'linear' | 'time-decay'

const MODELS: { id: ModelType; name: string; description: string }[] = [
  {
    id: 'first-touch',
    name: 'First Touch',
    description: '100% credit to first interaction',
  },
  {
    id: 'last-touch',
    name: 'Last Touch',
    description: '100% credit to final interaction',
  },
  {
    id: 'linear',
    name: 'Linear',
    description: 'Equal credit across all touchpoints',
  },
  {
    id: 'time-decay',
    name: 'Time Decay',
    description: 'More credit to recent interactions',
  },
]

export function AttributionDashboard() {
  const { sessions, persons, totalRevenue } = useStore()
  const [selectedModel, setSelectedModel] = useState<ModelType>('linear')

  const attributionData = useMemo(() => {
    const convertedSessions = sessions.filter((s) => s.converted && s.revenue)
    
    if (convertedSessions.length === 0) {
      return persons.map((p) => ({
        name: p.name,
        contribution: 0,
        revenue: 0,
        color: p.color,
        devices: p.devices.map((d) => ({ name: d.name, revenue: 0 })),
      }))
    }

    // Group sessions by person
    const personSessions: Record<string, typeof convertedSessions> = {}
    convertedSessions.forEach((session) => {
      if (!personSessions[session.personId]) {
        personSessions[session.personId] = []
      }
      personSessions[session.personId].push(session)
    })

    // Calculate attribution based on model
    const attribution: Record<string, number> = {}
    
    Object.entries(personSessions).forEach(([personId, personConversions]) => {
      attribution[personId] = 0

      personConversions.forEach((conversion) => {
        const conversionSessions = sessions.filter(
          (s) => s.personId === personId && s.timestamp <= conversion.timestamp
        )

        let weight = 0
        switch (selectedModel) {
          case 'first-touch':
            weight = conversionSessions[0]?.id === conversion.id ? 1 : 0
            break
          case 'last-touch':
            weight = 1
            break
          case 'linear':
            weight = 1 / conversionSessions.length
            break
          case 'time-decay':
            const index = conversionSessions.findIndex((s) => s.id === conversion.id)
            const decayFactor = Math.pow(0.8, conversionSessions.length - index - 1)
            weight = decayFactor / conversionSessions.reduce((sum, _, i) => 
              sum + Math.pow(0.8, conversionSessions.length - i - 1), 0
            )
            break
        }

        attribution[personId] += (conversion.revenue || 0) * weight
      })
    })

    // Calculate per-device attribution
    const deviceAttribution: Record<string, Record<string, number>> = {}
    convertedSessions.forEach((session) => {
      if (!deviceAttribution[session.personId]) {
        deviceAttribution[session.personId] = {}
      }
      if (!deviceAttribution[session.personId][session.deviceId]) {
        deviceAttribution[session.personId][session.deviceId] = 0
      }
      deviceAttribution[session.personId][session.deviceId] += session.revenue || 0
    })

    return persons.map((person) => ({
      name: person.name,
      contribution: totalRevenue > 0 ? (attribution[person.id] || 0) / totalRevenue : 0,
      revenue: attribution[person.id] || 0,
      color: person.color,
      devices: person.devices.map((device) => ({
        name: device.name,
        revenue: deviceAttribution[person.id]?.[device.id] || 0,
      })),
    }))
  }, [sessions, persons, selectedModel, totalRevenue])

  const chartData = attributionData.map((d) => ({
    name: d.name,
    revenue: d.revenue,
    contribution: d.contribution,
    fill: d.color,
  }))

  const pieData = attributionData.map((d) => ({
    name: d.name,
    value: d.revenue,
    color: d.color,
  }))

  const deviceData = useMemo(() => {
    const devices: Record<string, number> = {}
    sessions
      .filter((s) => s.converted && s.revenue)
      .forEach((session) => {
        const person = persons.find((p) => p.id === session.personId)
        const device = person?.devices.find((d) => d.id === session.deviceId)
        if (device) {
          devices[device.name] = (devices[device.name] || 0) + (session.revenue || 0)
        }
      })
    return Object.entries(devices).map(([name, revenue]) => ({
      name,
      revenue,
    }))
  }, [sessions, persons])

  const conversionTrend = useMemo(() => {
    const trend: Record<string, { conversions: number; revenue: number }> = {}
    sessions.forEach((session) => {
      const hour = new Date(session.timestamp).getHours()
      const key = `${hour}:00`
      if (!trend[key]) {
        trend[key] = { conversions: 0, revenue: 0 }
      }
      if (session.converted) {
        trend[key].conversions += 1
        trend[key].revenue += session.revenue || 0
      }
    })
    return Object.entries(trend)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .slice(-12)
      .map(([time, data]) => ({
        time,
        conversions: data.conversions,
        revenue: data.revenue,
      }))
  }, [sessions])

  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Attribution Dashboard
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Model Selector */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`p-3 rounded-lg text-left transition-colors ${
                selectedModel === model.id
                  ? 'bg-netflix-red text-white'
                  : 'bg-netflix-dark text-netflix-light hover:bg-gray-700'
              }`}
            >
              <div className="font-medium text-sm">{model.name}</div>
              <div className="text-xs opacity-80 mt-1">{model.description}</div>
            </button>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-netflix-dark rounded-lg p-4">
            <div className="text-netflix-gray text-xs mb-1">Total Revenue</div>
            <div className="text-2xl font-bold text-white">
              {formatCurrency(totalRevenue)}
            </div>
          </div>
          <div className="bg-netflix-dark rounded-lg p-4">
            <div className="text-netflix-gray text-xs mb-1">Conversions</div>
            <div className="text-2xl font-bold text-white">
              {sessions.filter((s) => s.converted).length}
            </div>
          </div>
          <div className="bg-netflix-dark rounded-lg p-4">
            <div className="text-netflix-gray text-xs mb-1">Avg Order Value</div>
            <div className="text-2xl font-bold text-white">
              {formatCurrency(
                sessions.filter((s) => s.converted).length > 0
                  ? totalRevenue / sessions.filter((s) => s.converted).length
                  : 0
              )}
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Revenue by Person */}
          <div className="bg-netflix-dark rounded-lg p-4">
            <h4 className="text-white font-medium mb-4">Revenue by Person</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="name" stroke="#808080" fontSize={12} />
                <YAxis
                  stroke="#808080"
                  fontSize={12}
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#181818',
                    border: '1px solid #333',
                    borderRadius: '4px',
                  }}
                  formatter={(value: number) => formatCurrency(value)}
                />
                <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Contribution Distribution */}
          <div className="bg-netflix-dark rounded-lg p-4">
            <h4 className="text-white font-medium mb-4">Contribution %</h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#181818',
                    border: '1px solid #333',
                    borderRadius: '4px',
                  }}
                  formatter={(value: number) => formatCurrency(value)}
                />
                <Legend
                  verticalAlign="bottom"
                  height={36}
                  formatter={(value) => (
                    <span style={{ color: '#fff' }}>{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Device Performance */}
        <div className="bg-netflix-dark rounded-lg p-4">
          <h4 className="text-white font-medium mb-4">Device Performance</h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={deviceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                type="number"
                stroke="#808080"
                fontSize={12}
                tickFormatter={(value) => `$${value}`}
              />
              <YAxis type="category" dataKey="name" stroke="#808080" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#181818',
                  border: '1px solid #333',
                  borderRadius: '4px',
                }}
                formatter={(value: number) => formatCurrency(value)}
              />
              <Bar dataKey="revenue" fill="#E50914" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Conversion Trend */}
        {conversionTrend.length > 0 && (
          <div className="bg-netflix-dark rounded-lg p-4">
            <h4 className="text-white font-medium mb-4">Conversion Trend</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={conversionTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#808080" fontSize={12} />
                <YAxis stroke="#808080" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#181818',
                    border: '1px solid #333',
                    borderRadius: '4px',
                  }}
                  formatter={(value: number, name: string) =>
                    name === 'revenue' ? formatCurrency(value) : value
                  }
                />
                <Line
                  type="monotone"
                  dataKey="conversions"
                  stroke="#E50914"
                  strokeWidth={2}
                  dot={{ fill: '#E50914' }}
                />
                <Line
                  type="monotone"
                  dataKey="revenue"
                  stroke="#00D4AA"
                  strokeWidth={2}
                  dot={{ fill: '#00D4AA' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
