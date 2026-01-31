'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ArchitectureViewProps {
  onClose: () => void
}

interface PipelineStage {
  id: string
  name: string
  icon: string
  status: 'active' | 'processing' | 'idle' | 'error'
  throughput: number
  latency: number
  details: {
    what: string
    input: string
    output: string
    technology: string
  }
}

const pipelineStages: PipelineStage[] = [
  {
    id: 'ingestion',
    name: 'Event Ingestion',
    icon: '📥',
    status: 'active',
    throughput: 24583,
    latency: 12,
    details: {
      what: 'Captures and validates all incoming touchpoint events from websites, apps, and external sources.',
      input: 'Raw events (page views, clicks, conversions) from SDKs, webhooks, and APIs',
      output: 'Validated, normalized events in canonical schema',
      technology: 'Kafka Topics (24 partitions), JSON Schema validation, Circuit breaker pattern',
    },
  },
  {
    id: 'features',
    name: 'Feature Extraction',
    icon: '🔍',
    status: 'active',
    throughput: 24580,
    latency: 28,
    details: {
      what: 'Extracts 47+ behavioral and technical features from each event for identity resolution.',
      input: 'Normalized events from ingestion layer',
      output: 'Feature vectors with 47+ dimensions per event',
      technology: 'Apache Flink Stateful Stream Processing, 15min tumbling windows',
    },
  },
  {
    id: 'clustering',
    name: 'ML Clustering',
    icon: '🧠',
    status: 'processing',
    throughput: 12456,
    latency: 45,
    details: {
      what: 'Machine learning models group similar events into identity clusters using ensemble methods.',
      input: 'Feature vectors and historical cluster data',
      output: 'Cluster assignments with confidence scores',
      technology: 'XGBoost + Isolation Forest ensemble, TensorFlow Serving, <50ms inference',
    },
  },
  {
    id: 'assignment',
    name: 'Identity Assignment',
    icon: '🆔',
    status: 'active',
    throughput: 12450,
    latency: 8,
    details: {
      what: 'Maps clusters to persistent identities and updates the identity graph.',
      input: 'Cluster assignments and confidence scores',
      output: 'Resolved identities, updated identity graph',
      technology: 'PostgreSQL + TimescaleDB, Redis cache layer, Graph database (Neo4j)',
    },
  },
  {
    id: 'attribution',
    name: 'Attribution Engine',
    icon: '⚖️',
    status: 'active',
    throughput: 3421,
    latency: 15,
    details: {
      what: 'Calculates multi-touch attribution across the customer journey using various models.',
      input: 'Resolved identities and conversion events',
      output: 'Attribution weights, journey analytics, ROI metrics',
      technology: 'Shapley value computation, CUPED variance reduction, 30-90d lookback',
    },
  },
]

interface InfrastructureComponent {
  id: string
  name: string
  type: 'cache' | 'database' | 'api' | 'model'
  status: 'healthy' | 'degraded' | 'down'
  metrics: { label: string; value: string }[]
}

const infrastructure: InfrastructureComponent[] = [
  {
    id: 'redis',
    name: 'Redis Cache',
    type: 'cache',
    status: 'healthy',
    metrics: [
      { label: 'Hit Rate', value: '94.2%' },
      { label: 'Memory', value: '12.4GB' },
      { label: 'Latency', value: '0.8ms' },
    ],
  },
  {
    id: 'postgres',
    name: 'PostgreSQL',
    type: 'database',
    status: 'healthy',
    metrics: [
      { label: 'TPS', value: '4.2K' },
      { label: 'Connections', value: '142' },
      { label: 'Lag', value: '0ms' },
    ],
  },
  {
    id: 'fastapi',
    name: 'FastAPI Cluster',
    type: 'api',
    status: 'healthy',
    metrics: [
      { label: 'RPS', value: '12.5K' },
      { label: 'P99 Latency', value: '45ms' },
      { label: 'Instances', value: '24' },
    ],
  },
  {
    id: 'ml-models',
    name: 'ML Models',
    type: 'model',
    status: 'healthy',
    metrics: [
      { label: 'Accuracy', value: '94.7%' },
      { label: 'Inferences/s', value: '25K' },
      { label: 'AUC-ROC', value: '0.967' },
    ],
  },
]

export function ArchitectureView({ onClose }: ArchitectureViewProps) {
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null)
  const [selectedInfra, setSelectedInfra] = useState<InfrastructureComponent | null>(null)
  const [flowAnimation, setFlowAnimation] = useState(0)
  const [activeParticles, setActiveParticles] = useState<Array<{ id: number; stage: number; progress: number }>>([])
  const containerRef = useRef<HTMLDivElement>(null)

  // Flow animation
  useEffect(() => {
    const interval = setInterval(() => {
      setFlowAnimation(prev => (prev + 1) % 4)
    }, 800)
    return () => clearInterval(interval)
  }, [])

  // Particle animation
  useEffect(() => {
    const createParticle = () => {
      const newParticle = {
        id: Date.now() + Math.random(),
        stage: 0,
        progress: 0,
      }
      setActiveParticles(prev => [...prev.slice(-20), newParticle])
    }

    const interval = setInterval(createParticle, 400)
    return () => clearInterval(interval)
  }, [])

  // Update particle positions
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveParticles(prev =>
        prev
          .map(p => ({
            ...p,
            progress: p.progress + 0.1,
            stage: p.progress >= 1 ? p.stage + 1 : p.stage,
          }))
          .filter(p => p.stage < pipelineStages.length)
          .map(p => ({ ...p, progress: p.progress >= 1 ? 0 : p.progress }))
      )
    }, 50)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'healthy':
        return '#00ff41'
      case 'processing':
        return '#ffb800'
      case 'degraded':
        return '#ff6b35'
      case 'error':
      case 'down':
        return '#ff3333'
      default:
        return '#606060'
    }
  }

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'active':
      case 'healthy':
        return 'status-operational'
      case 'processing':
        return 'status-warning'
      case 'degraded':
        return 'status-warning'
      case 'error':
      case 'down':
        return 'status-critical'
      default:
        return 'status-info'
    }
  }

  return (
    <div className="tactical-card" data-tour="architecture">
      {/* Header */}
      <div className="tactical-header border-b-2 border-[#404040]">
        <div className="flex items-center gap-2">
          <span className="text-[#00ff41] text-lg">⚡</span>
          <span className="text-sm font-bold tracking-wider uppercase text-[#e0e0e0]">
            System Architecture
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-xs font-mono text-[#606060]">
            Live Data Flow
          </div>
          <div className="w-2 h-2 rounded-full bg-[#00ff41] animate-pulse" />
          <button
            onClick={onClose}
            className="text-[#606060] hover:text-[#a0a0a0] text-xl ml-2"
          >
            ×
          </button>
        </div>
      </div>

      <div ref={containerRef} className="p-6 space-y-8">
        {/* Pipeline Flow */}
        <div className="relative">
          {/* Connecting lines */}
          <div className="absolute top-[60px] left-[10%] right-[10%] h-0.5 bg-[#2d2d2d]">
            {pipelineStages.slice(0, -1).map((_, index) => (
              <div
                key={index}
                className="absolute h-full bg-gradient-to-r from-transparent via-[#00ff41] to-transparent transition-all duration-500"
                style={{
                  left: `${(index / (pipelineStages.length - 1)) * 100}%`,
                  width: `${100 / (pipelineStages.length - 1)}%`,
                  opacity: flowAnimation === index ? 1 : 0.2,
                }}
              />
            ))}
          </div>

          {/* Data flow arrows */}
          <div className="absolute top-[56px] left-[10%] right-[10%] flex justify-between px-8">
            {pipelineStages.slice(0, -1).map((_, index) => (
              <motion.div
                key={index}
                className="text-[#00ff41]/50"
                animate={{ x: [0, 5, 0], opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: index * 0.2 }}
              >
                →
              </motion.div>
            ))}
          </div>

          {/* Pipeline stages */}
          <div className="relative z-10 grid grid-cols-5 gap-2">
            {pipelineStages.map((stage, index) => (
              <motion.button
                key={stage.id}
                className={`relative group ${selectedStage?.id === stage.id ? 'z-20' : ''}`}
                onClick={() => setSelectedStage(selectedStage?.id === stage.id ? null : stage)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {/* Particle effects */}
                <div className="absolute inset-0 overflow-hidden rounded-sm pointer-events-none">
                  {activeParticles
                    .filter(p => p.stage === index)
                    .map(particle => (
                      <motion.div
                        key={particle.id}
                        className="absolute w-1.5 h-1.5 rounded-full bg-[#00ff41]"
                        style={{
                          top: '50%',
                          left: '0%',
                          transform: 'translate(-50%, -50%)',
                        }}
                        animate={{
                          left: `${particle.progress * 100}%`,
                          opacity: [1, 1, 0],
                        }}
                        transition={{ duration: 0.5, ease: 'linear' }}
                      />
                    ))}
                </div>

                <div
                  className={`tactical-card p-3 cursor-pointer transition-all duration-300 ${
                    selectedStage?.id === stage.id
                      ? 'border-[#00ff41] shadow-[0_0_30px_rgba(0,255,65,0.3)]'
                      : 'hover:border-[#00ff41]/50'
                  }`}
                >
                  {/* Status indicator */}
                  <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-[#1a1a1a]"
                    style={{ backgroundColor: getStatusColor(stage.status) }}
                  />

                  <div className="text-center space-y-2">
                    <div className="text-2xl">{stage.icon}</div>
                    <div className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider">
                      {stage.name}
                    </div>
                    <div className={`status-indicator text-[10px] py-0.5 px-2 ${getStatusClass(stage.status)}`}>
                      {stage.status}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="mt-3 space-y-1 pt-3 border-t border-[#404040]/50">
                    <div className="flex justify-between text-[10px] font-mono">
                      <span className="text-[#606060]">Throughput</span>
                      <span className="text-[#00ff41]">{stage.throughput.toLocaleString()}/s</span>
                    </div>
                    <div className="flex justify-between text-[10px] font-mono">
                      <span className="text-[#606060]">Latency</span>
                      <span className="text-[#00bfff]">{stage.latency}ms</span>
                    </div>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>

          {/* Stage detail panel */}
          <AnimatePresence>
            {selectedStage && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-full left-0 right-0 mt-4 tactical-card z-30"
              >
                <div className="tactical-header border-b border-[#404040]">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{selectedStage.icon}</span>
                    <span className="font-bold text-[#e0e0e0]">{selectedStage.name}</span>
                  </div>
                  <button
                    onClick={() => setSelectedStage(null)}
                    className="text-[#606060] hover:text-[#a0a0a0] text-xl"
                  >
                    ×
                  </button>
                </div>
                <div className="p-4 grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="text-xs font-bold text-[#00ff41] uppercase">Description</div>
                    <p className="text-xs text-[#a0a0a0]">{selectedStage.details.what}</p>
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs font-bold text-[#00bfff] uppercase">Data Flow</div>
                    <div className="text-xs font-mono space-y-1">
                      <div className="text-[#606060]">IN: <span className="text-[#e0e0e0]">{selectedStage.details.input}</span></div>
                      <div className="text-[#606060]">OUT: <span className="text-[#e0e0e0]">{selectedStage.details.output}</span></div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs font-bold text-[#ffb800] uppercase">Technology</div>
                    <p className="text-xs text-[#a0a0a0] font-mono">{selectedStage.details.technology}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Infrastructure layer */}
        <div className="border-t-2 border-[#404040] pt-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-[#00bfff] text-sm">🔧</span>
            <span className="text-xs font-bold tracking-wider uppercase text-[#a0a0a0]">
              Infrastructure Layer
            </span>
          </div>

          <div className="grid grid-cols-4 gap-3">
            {infrastructure.map((infra) => (
              <button
                key={infra.id}
                onClick={() => setSelectedInfra(selectedInfra?.id === infra.id ? null : infra)}
                className={`tactical-card p-3 text-left transition-all ${
                  selectedInfra?.id === infra.id
                    ? 'border-[#00bfff] shadow-[0_0_20px_rgba(0,191,255,0.2)]'
                    : 'hover:border-[#404040]/80'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-[#e0e0e0]">{infra.name}</span>
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getStatusColor(infra.status) }}
                  />
                </div>

                <div className="space-y-1">
                  {infra.metrics.map((metric, idx) => (
                    <div key={idx} className="flex justify-between text-[10px] font-mono">
                      <span className="text-[#606060]">{metric.label}</span>
                      <span className="text-[#e0e0e0]">{metric.value}</span>
                    </div>
                  ))}
                </div>

                {selectedInfra?.id === infra.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 pt-3 border-t border-[#404040]/50"
                  >
                    <div className="text-[10px] font-mono text-[#00bfff]">
                      Status: <span className="text-[#e0e0e0] uppercase">{infra.status}</span>
                    </div>
                  </motion.div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 text-[10px] font-mono text-[#606060]">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#00ff41]" />
            <span>Active/Healthy</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#ffb800]" />
            <span>Processing/Degraded</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#ff3333]" />
            <span>Error/Down</span>
          </div>
          <div className="flex items-center gap-2 border-l border-[#404040] pl-6">
            <span>Click any component for details</span>
          </div>
        </div>
      </div>
    </div>
  )
}
