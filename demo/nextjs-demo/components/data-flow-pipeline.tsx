'use client'

import { useState, useEffect } from 'react'

interface PipelineStage {
  id: string
  name: string
  status: 'active' | 'processing' | 'idle' | 'error'
  throughput: number
  latency: number
}

interface DataPacket {
  id: string
  stage: number
  progress: number
}

export function DataFlowPipeline() {
  const [stages, setStages] = useState<PipelineStage[]>([
    { id: 'ingest', name: 'RAW EVENTS', status: 'active', throughput: 1250, latency: 12 },
    { id: 'extract', name: 'FEATURE EXTRACTOR', status: 'active', throughput: 1180, latency: 24 },
    { id: 'cluster', name: 'CLUSTERING ENGINE', status: 'processing', throughput: 980, latency: 45 },
    { id: 'assign', name: 'ASSIGNMENT OUTPUT', status: 'active', throughput: 975, latency: 8 },
  ])

  const [packets, setPackets] = useState<DataPacket[]>([])
  const [totalProcessed, setTotalProcessed] = useState(45678)
  const [uptime] = useState(() => Math.floor(Math.random() * 999) + 1000)

  // Simulate stage updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStages(prev => prev.map(stage => ({
        ...stage,
        throughput: stage.throughput + Math.floor(Math.random() * 50 - 25),
        latency: Math.max(5, stage.latency + Math.floor(Math.random() * 6 - 3)),
        status: Math.random() > 0.9 ? 'processing' : 'active',
      })))

      setTotalProcessed(prev => prev + Math.floor(Math.random() * 20 + 10))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // Generate data packets
  useEffect(() => {
    const interval = setInterval(() => {
      const newPacket: DataPacket = {
        id: Math.random().toString(36).substring(2, 8).toUpperCase(),
        stage: 0,
        progress: 0,
      }
      setPackets(prev => [...prev, newPacket])
    }, 800)

    return () => clearInterval(interval)
  }, [])

  // Animate packets through stages
  useEffect(() => {
    const interval = setInterval(() => {
      setPackets(prev => {
        return prev
          .map(packet => {
            const newProgress = packet.progress + 2
            if (newProgress >= 100) {
              return {
                ...packet,
                stage: Math.min(packet.stage + 1, stages.length - 1),
                progress: 0,
              }
            }
            return { ...packet, progress: newProgress }
          })
          .filter(packet => packet.stage < stages.length)
      })
    }, 50)

    return () => clearInterval(interval)
  }, [stages.length])

  const getStatusColor = (status: PipelineStage['status']) => {
    switch (status) {
      case 'active': return 'bg-[#00ff41]'
      case 'processing': return 'bg-[#ffb800]'
      case 'idle': return 'bg-[#606060]'
      case 'error': return 'bg-[#ff3333]'
      default: return 'bg-[#606060]'
    }
  }

  const getStatusText = (status: PipelineStage['status']) => {
    switch (status) {
      case 'active': return 'ACTIVE'
      case 'processing': return 'PROCESSING'
      case 'idle': return 'IDLE'
      case 'error': return 'ERROR'
      default: return 'UNKNOWN'
    }
  }

  return (
    <div className="tactical-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-[#00ff41] rounded-full animate-pulse"></div>
          <span className="text-sm font-bold tracking-widest text-[#e0e0e0]">
            DATA FLOW PIPELINE
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <div className="text-[#606060] uppercase">Processed</div>
            <div className="text-[#00ff41] font-bold">{totalProcessed.toLocaleString()}</div>
          </div>
          <div className="text-right">
            <div className="text-[#606060] uppercase">Uptime</div>
            <div className="text-[#e0e0e0]">{uptime}s</div>
          </div>
        </div>
      </div>

      {/* Pipeline Visualization */}
      <div className="relative">
        {/* Connection Lines */}
        <div className="absolute top-[45px] left-[12.5%] right-[12.5%] h-[2px] flex">
          {[0, 1, 2].map(i => (
            <div key={i} className="flex-1 flex items-center">
              <div className="h-[2px] flex-1 bg-gradient-to-r from-[#00ff41]/50 to-[#00ff41]/20"></div>
              <div className="w-2 h-2 rotate-45 border-r-2 border-t-2 border-[#00ff41]/50"></div>
            </div>
          ))}
        </div>

        {/* Data Packets */}
        <div className="absolute top-[40px] left-[12.5%] right-[12.5%] h-[12px] pointer-events-none">
          {packets.map(packet => {
            const stageWidth = 100 / (stages.length - 1)
            const leftPos = packet.stage * stageWidth + (packet.progress / 100) * stageWidth
            return (
              <div
                key={packet.id}
                className="absolute w-3 h-3 bg-[#00ff41] rounded-sm shadow-[0_0_10px_rgba(0,255,65,0.8)] transition-all duration-100"
                style={{ 
                  left: `calc(${leftPos}% - 6px)`,
                  transform: 'rotate(45deg)'
                }}
              ></div>
            )
          })}
        </div>

        {/* Stages */}
        <div className="flex justify-between relative z-10">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex flex-col items-center w-[22%]">
              {/* Stage Box */}
              <div className="w-full border border-[#404040] bg-[#1a1a1a] p-3 relative">
                {/* Corner Accents */}
                <div className="absolute top-0 left-0 w-2 h-2 border-l-2 border-t-2 border-[#00ff41]/50"></div>
                <div className="absolute top-0 right-0 w-2 h-2 border-r-2 border-t-2 border-[#00ff41]/50"></div>
                <div className="absolute bottom-0 left-0 w-2 h-2 border-l-2 border-b-2 border-[#00ff41]/50"></div>
                <div className="absolute bottom-0 right-0 w-2 h-2 border-r-2 border-b-2 border-[#00ff41]/50"></div>

                {/* Stage Content */}
                <div className="text-center">
                  <div className="text-[10px] text-[#606060] mb-1 tracking-wider">
                    STAGE {index + 1}
                  </div>
                  <div className="text-xs font-bold text-[#e0e0e0] mb-2 tracking-wide">
                    {stage.name}
                  </div>
                  
                  {/* Status Indicator */}
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(stage.status)} animate-pulse`}></div>
                    <span className={`text-[10px] font-bold ${
                      stage.status === 'active' ? 'text-[#00ff41]' : 
                      stage.status === 'processing' ? 'text-[#ffb800]' : 
                      stage.status === 'error' ? 'text-[#ff3333]' : 'text-[#606060]'
                    }`}>
                      {getStatusText(stage.status)}
                    </span>
                  </div>

                  {/* Metrics */}
                  <div className="space-y-1 text-[10px] font-mono">
                    <div className="flex justify-between">
                      <span className="text-[#606060]">TPS:</span>
                      <span className="text-[#00bfff]">{stage.throughput.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#606060]">LATENCY:</span>
                      <span className="text-[#ffb800]">{stage.latency}ms</span>
                    </div>
                  </div>
                </div>

                {/* Processing Animation */}
                {stage.status === 'processing' && (
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#ffb800]/10 to-transparent animate-[processing_2s_linear_infinite]"></div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline Stats */}
      <div className="mt-6 pt-4 border-t border-[#404040]">
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-[10px] text-[#606060] uppercase tracking-wider mb-1">Queue Depth</div>
            <div className="text-lg font-bold text-[#00ff41] font-mono">142</div>
          </div>
          <div>
            <div className="text-[10px] text-[#606060] uppercase tracking-wider mb-1">Error Rate</div>
            <div className="text-lg font-bold text-[#00bfff] font-mono">0.02%</div>
          </div>
          <div>
            <div className="text-[10px] text-[#606060] uppercase tracking-wider mb-1">Avg Latency</div>
            <div className="text-lg font-bold text-[#ffb800] font-mono">22ms</div>
          </div>
          <div>
            <div className="text-[10px] text-[#606060] uppercase tracking-wider mb-1">Throughput</div>
            <div className="text-lg font-bold text-[#e0e0e0] font-mono">975/s</div>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="mt-4 pt-3 border-t border-[#404040] flex items-center justify-between text-[10px] font-mono text-[#606060]">
        <div className="flex items-center gap-4">
          <span>PIPELINE_ID: PIPE-8472-X</span>
          <span>VERSION: 2.4.1</span>
          <span>NODES: 4/4 ACTIVE</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[#00ff41]">●</span>
          <span className="text-[#00ff41]">ALL SYSTEMS OPERATIONAL</span>
        </div>
      </div>
    </div>
  )
}
