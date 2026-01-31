'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'

interface Node3D {
  id: string
  type: 'person' | 'device' | 'session'
  label: string
  color: string
  x: number
  y: number
  z: number
  size: number
  connections: string[]
  metadata?: {
    confidence?: number
    deviceType?: string
    lastActivity?: string
  }
}

interface Cluster3D {
  id: string
  name: string
  center: { x: number; y: number; z: number }
  nodes: Node3D[]
  color: string
  radius: number
}

// Generate sample 3D data
const generate3DClusters = (): Cluster3D[] => {
  const clusters: Cluster3D[] = [
    {
      id: 'cluster-1',
      name: 'Household A',
      center: { x: 0, y: 0, z: 0 },
      color: '#00ff41',
      radius: 120,
      nodes: [
        { id: 'p1', type: 'person', label: 'Alice', color: '#00ff41', x: -40, y: -20, z: 20, size: 16, connections: ['d1', 'd2', 's1', 's2'], metadata: { confidence: 0.96 } },
        { id: 'p2', type: 'person', label: 'Bob', color: '#00ff41', x: 40, y: 30, z: -10, size: 16, connections: ['d3', 's3'], metadata: { confidence: 0.94 } },
        { id: 'd1', type: 'device', label: 'iPhone 14', color: '#00bfff', x: -60, y: 10, z: 40, size: 10, connections: ['p1', 's1'], metadata: { deviceType: 'mobile' } },
        { id: 'd2', type: 'device', label: 'MacBook Pro', color: '#00bfff', x: -20, y: -50, z: 30, size: 10, connections: ['p1', 's2'], metadata: { deviceType: 'desktop' } },
        { id: 'd3', type: 'device', label: 'iPad Air', color: '#00bfff', x: 50, y: 50, z: -30, size: 10, connections: ['p2', 's3'], metadata: { deviceType: 'tablet' } },
        { id: 's1', type: 'session', label: 'S-001', color: '#ffb800', x: -70, y: 20, z: 60, size: 6, connections: ['p1', 'd1'], metadata: { lastActivity: '2 min ago' } },
        { id: 's2', type: 'session', label: 'S-002', color: '#ffb800', x: -30, y: -60, z: 50, size: 6, connections: ['p1', 'd2'], metadata: { lastActivity: '15 min ago' } },
        { id: 's3', type: 'session', label: 'S-003', color: '#ffb800', x: 60, y: 60, z: -50, size: 6, connections: ['p2', 'd3'], metadata: { lastActivity: '1 hour ago' } },
      ],
    },
    {
      id: 'cluster-2',
      name: 'Household B',
      center: { x: 200, y: 100, z: -80 },
      color: '#ff6b35',
      radius: 100,
      nodes: [
        { id: 'p3', type: 'person', label: 'Carol', color: '#ff6b35', x: 180, y: 80, z: -60, size: 16, connections: ['d4', 'd5', 's4'], metadata: { confidence: 0.91 } },
        { id: 'd4', type: 'device', label: 'Galaxy S23', color: '#00bfff', x: 160, y: 120, z: -40, size: 10, connections: ['p3', 's4'], metadata: { deviceType: 'mobile' } },
        { id: 'd5', type: 'device', label: 'Windows PC', color: '#00bfff', x: 220, y: 70, z: -100, size: 10, connections: ['p3', 's5'], metadata: { deviceType: 'desktop' } },
        { id: 's4', type: 'session', label: 'S-004', color: '#ffb800', x: 140, y: 140, z: -20, size: 6, connections: ['p3', 'd4'], metadata: { lastActivity: '5 min ago' } },
        { id: 's5', type: 'session', label: 'S-005', color: '#ffb800', x: 240, y: 50, z: -120, size: 6, connections: ['p3', 'd5'], metadata: { lastActivity: '30 min ago' } },
      ],
    },
    {
      id: 'cluster-3',
      name: 'Cross-Household Match',
      center: { x: -150, y: 80, z: 100 },
      color: '#00bfff',
      radius: 90,
      nodes: [
        { id: 'p4', type: 'person', label: 'David', color: '#00bfff', x: -170, y: 60, z: 120, size: 16, connections: ['d6', 's6'], metadata: { confidence: 0.87 } },
        { id: 'd6', type: 'device', label: 'Chromebook', color: '#00bfff', x: -130, y: 100, z: 80, size: 10, connections: ['p4', 's6'], metadata: { deviceType: 'laptop' } },
        { id: 's6', type: 'session', label: 'S-006', color: '#ffb800', x: -110, y: 120, z: 60, size: 6, connections: ['p4', 'd6'], metadata: { lastActivity: '10 min ago' } },
      ],
    },
  ]

  return clusters
}

export function Enhanced3DViz() {
  const [clusters] = useState(() => generate3DClusters())
  const [rotation, setRotation] = useState({ x: -20, y: 45 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [hoveredNode, setHoveredNode] = useState<Node3D | null>(null)
  const [selectedNode, setSelectedNode] = useState<Node3D | null>(null)
  const [autoRotate, setAutoRotate] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()

  // Auto-rotation
  useEffect(() => {
    if (autoRotate && !isDragging) {
      const animate = () => {
        setRotation(prev => ({
          x: prev.x,
          y: (prev.y + 0.5) % 360,
        }))
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }
  }, [autoRotate, isDragging])

  // Transform 3D coordinates to 2D screen space
  const project3D = useCallback((x: number, y: number, z: number, baseRotation = rotation) => {
    const radX = (baseRotation.x * Math.PI) / 180
    const radY = (baseRotation.y * Math.PI) / 180

    // Rotate around Y axis
    let x1 = x * Math.cos(radY) - z * Math.sin(radY)
    let z1 = x * Math.sin(radY) + z * Math.cos(radY)

    // Rotate around X axis
    let y1 = y * Math.cos(radX) - z1 * Math.sin(radX)
    let z2 = y * Math.sin(radX) + z1 * Math.cos(radX)

    // Perspective projection
    const perspective = 800
    const scale = perspective / (perspective + z2)

    return {
      x: x1 * scale,
      y: y1 * scale,
      z: z2,
      scale,
    }
  }, [rotation])

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX, y: e.clientY })
    setAutoRotate(false)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return

    const deltaX = e.clientX - dragStart.x
    const deltaY = e.clientY - dragStart.y

    setRotation(prev => ({
      x: Math.max(-60, Math.min(60, prev.x - deltaY * 0.5)),
      y: (prev.y + deltaX * 0.5) % 360,
    }))

    setDragStart({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'person':
        return '👤'
      case 'device':
        return '💻'
      case 'session':
        return '⚡'
      default:
        return '•'
    }
  }

  const getDepthShade = (z: number) => {
    // Darker when further back
    const normalizedZ = (z + 200) / 400 // Normalize to 0-1 range
    const brightness = 0.5 + normalizedZ * 0.5 // 0.5 to 1.0
    return brightness
  }

  // Get all connections for rendering
  const getAllConnections = () => {
    const connections: Array<{ from: Node3D; to: Node3D }> = []
    const allNodes = clusters.flatMap(c => c.nodes)
    const nodeMap = new Map(allNodes.map(n => [n.id, n]))

    allNodes.forEach(node => {
      node.connections.forEach(targetId => {
        const target = nodeMap.get(targetId)
        if (target && node.id < target.id) { // Avoid duplicates
          connections.push({ from: node, to: target })
        }
      })
    })

    return connections
  }

  const connections = getAllConnections()
  const allNodes = clusters.flatMap(c => c.nodes)

  // Sort by Z depth for proper rendering order
  const sortedNodes = [...allNodes].sort((a, b) => {
    const aProj = project3D(a.x, a.y, a.z)
    const bProj = project3D(b.x, b.y, b.z)
    return bProj.z - aProj.z
  })

  return (
    <div className="tactical-card" data-tour="clustering-3d">
      {/* Header */}
      <div className="tactical-header border-b-2 border-[#404040]">
        <div className="flex items-center gap-2">
          <span className="text-[#00ff41] text-lg">🎯</span>
          <span className="text-sm font-bold tracking-wider uppercase text-[#e0e0e0]">
            3D Identity Clustering
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`tactical-btn text-xs py-1 px-3 ${autoRotate ? 'tactical-btn-primary' : ''}`}
          >
            {autoRotate ? '⏹ Stop' : '▶ Auto'}
          </button>
          <button
            onClick={() => setRotation({ x: -20, y: 45 })}
            className="tactical-btn text-xs py-1 px-3"
          >
            ↺ Reset
          </button>
        </div>
      </div>

      <div className="p-4">
        {/* 3D Viewport */}
        <div
          ref={containerRef}
          className="relative h-[400px] bg-[#0f0f0f] rounded-sm overflow-hidden cursor-move select-none"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            perspective: '1000px',
            perspectiveOrigin: 'center center',
          }}
        >
          {/* Grid lines for depth reference */}
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage: `
                linear-gradient(to right, #00ff41 1px, transparent 1px),
                linear-gradient(to bottom, #00ff41 1px, transparent 1px)
              `,
              backgroundSize: '50px 50px',
              transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
              transformStyle: 'preserve-3d',
            }}
          />

          {/* 3D Scene */}
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
              transformStyle: 'preserve-3d',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out',
            }}
          >
            {/* Cluster bounds */}
            {clusters.map((cluster) => {
              const center = project3D(cluster.center.x, cluster.center.y, cluster.center.z)
              const brightness = getDepthShade(center.z)

              return (
                <div
                  key={cluster.id}
                  className="absolute rounded-full border-2 border-dashed"
                  style={{
                    width: cluster.radius * 2 * center.scale,
                    height: cluster.radius * 2 * center.scale,
                    left: `calc(50% + ${center.x}px - ${cluster.radius * center.scale}px)`,
                    top: `calc(50% + ${center.y}px - ${cluster.radius * center.scale}px)`,
                    borderColor: cluster.color,
                    opacity: 0.3 * brightness,
                    boxShadow: `0 0 ${30 * brightness}px ${cluster.color}40`,
                    transform: `translateZ(${center.z}px)`,
                  }}
                >
                  <div
                    className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold whitespace-nowrap"
                    style={{ color: cluster.color }}
                  >
                    {cluster.name}
                  </div>
                </div>
              )
            })}

            {/* Connections */}
            {connections.map((conn, idx) => {
              const from = project3D(conn.from.x, conn.from.y, conn.from.z)
              const to = project3D(conn.to.x, conn.to.y, conn.to.z)
              const avgZ = (from.z + to.z) / 2
              const brightness = getDepthShade(avgZ)

              const dx = to.x - from.x
              const dy = to.y - from.y
              const length = Math.sqrt(dx * dx + dy * dy)
              const angle = (Math.atan2(dy, dx) * 180) / Math.PI

              return (
                <div
                  key={idx}
                  className="absolute h-0.5 origin-left"
                  style={{
                    width: length,
                    left: `calc(50% + ${from.x}px)`,
                    top: `calc(50% + ${from.y}px)`,
                    background: `linear-gradient(to right, ${conn.from.color}80, ${conn.to.color}80)`,
                    transform: `translateZ(${avgZ}px) rotate(${angle}deg)`,
                    opacity: 0.6 * brightness,
                  }}
                />
              )
            })}

            {/* Nodes */}
            {sortedNodes.map((node) => {
              const projected = project3D(node.x, node.y, node.z)
              const brightness = getDepthShade(projected.z)
              const isHovered = hoveredNode?.id === node.id
              const isSelected = selectedNode?.id === node.id
              const isConnected = selectedNode?.connections.includes(node.id) ||
                                  selectedNode && node.connections.includes(selectedNode.id)

              return (
                <motion.div
                  key={node.id}
                  className="absolute rounded-full flex items-center justify-center cursor-pointer"
                  style={{
                    width: node.size * projected.scale * (isHovered || isSelected ? 1.3 : 1),
                    height: node.size * projected.scale * (isHovered || isSelected ? 1.3 : 1),
                    left: `calc(50% + ${projected.x}px)`,
                    top: `calc(50% + ${projected.y}px)`,
                    backgroundColor: node.color,
                    transform: `translate(-50%, -50%) translateZ(${projected.z}px)`,
                    boxShadow: isSelected
                      ? `0 0 30px ${node.color}, 0 0 60px ${node.color}80`
                      : isConnected
                      ? `0 0 20px ${node.color}60`
                      : `0 0 ${10 * brightness}px ${node.color}40`,
                    opacity: brightness,
                    filter: selectedNode && !isSelected && !isConnected ? 'brightness(0.4)' : 'none',
                  }}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                  onClick={() => setSelectedNode(selectedNode?.id === node.id ? null : node)}
                  whileHover={{ scale: 1.2 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  <span className="text-xs" style={{ opacity: brightness }}>
                    {getNodeIcon(node.type)}
                  </span>
                </motion.div>
              )
            })}
          </div>

          {/* HUD Overlay */}
          <div className="absolute top-4 left-4 space-y-2 pointer-events-none">
            <div className="text-xs font-mono text-[#00ff41]">
              ROTATION: X:{rotation.x.toFixed(0)}° Y:{rotation.y.toFixed(0)}°
            </div>
            <div className="text-xs font-mono text-[#606060]">
              NODES: {allNodes.length} | CLUSTERS: {clusters.length}
            </div>
          </div>

          {/* Instructions */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end pointer-events-none">
            <div className="text-xs font-mono text-[#606060]">
              Drag to rotate • Click nodes for details
            </div>
            <div className="text-xs font-mono text-[#00bfff]">
              {isDragging ? 'ROTATING...' : autoRotate ? 'AUTO-ROTATING' : 'READY'}
            </div>
          </div>

          {/* Hover tooltip */}
          {hoveredNode && !selectedNode && (
            <div
              className="absolute pointer-events-none tactical-card px-3 py-2 z-10"
              style={{
                left: '50%',
                top: '20px',
                transform: 'translateX(-50%)',
              }}
            >
              <div className="text-xs font-bold text-[#e0e0e0]">{hoveredNode.label}</div>
              <div className="text-[10px] text-[#606060] uppercase">{hoveredNode.type}</div>
            </div>
          )}
        </div>

        {/* Detail Panel */}
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 tactical-card p-4"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-xl"
                  style={{ backgroundColor: selectedNode.color }}
                >
                  {getNodeIcon(selectedNode.type)}
                </div>
                <div>
                  <div className="font-bold text-[#e0e0e0]">{selectedNode.label}</div>
                  <div className="text-xs text-[#606060] uppercase">{selectedNode.type} Node</div>
                </div>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-[#606060] hover:text-[#a0a0a0] text-xl"
              >
                ×
              </button>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-4">
              <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
                <div className="text-[10px] text-[#606060] uppercase">Connections</div>
                <div className="text-lg font-bold text-[#00ff41]">{selectedNode.connections.length}</div>
              </div>

              {selectedNode.metadata?.confidence && (
                <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
                  <div className="text-[10px] text-[#606060] uppercase">Confidence</div>
                  <div className="text-lg font-bold text-[#00bfff]">
                    {(selectedNode.metadata.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {selectedNode.metadata?.deviceType && (
                <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
                  <div className="text-[10px] text-[#606060] uppercase">Device Type</div>
                  <div className="text-lg font-bold text-[#ffb800]">{selectedNode.metadata.deviceType}</div>
                </div>
              )}

              {selectedNode.metadata?.lastActivity && (
                <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3">
                  <div className="text-[10px] text-[#606060] uppercase">Last Activity</div>
                  <div className="text-lg font-bold text-[#e0e0e0]">{selectedNode.metadata.lastActivity}</div>
                </div>
              )}
            </div>

            <div className="mt-4 text-xs text-[#a0a0a0]">
              <span className="text-[#00ff41]">Connected to:</span>{' '}
              {selectedNode.connections.map(id => {
                const connected = allNodes.find(n => n.id === id)
                return connected?.label
              }).filter(Boolean).join(', ')}
            </div>
          </motion.div>
        )}

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#00ff41]" />
            <span className="text-xs text-[#a0a0a0]">Person (Identity)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#00bfff]" />
            <span className="text-xs text-[#a0a0a0]">Device</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#ffb800]" />
            <span className="text-xs text-[#a0a0a0]">Session</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-gradient-to-r from-[#00ff41] to-[#00ff41]/30" />
            <span className="text-xs text-[#a0a0a0]">Connection</span>
          </div>
        </div>

        {/* Educational note */}
        <div className="mt-4 alert-box alert-box-info py-2 px-3">
          <div className="flex items-start gap-2">
            <span className="text-[#00bfff]">ℹ</span>
            <div>
              <div className="text-xs font-bold text-[#00bfff] uppercase mb-1">Understanding the 3D View</div>
              <p className="text-xs text-[#a0a0a0]">
                Depth (Z-axis) represents confidence and temporal distance. Nodes further back (darker) 
                have lower confidence scores or occurred earlier. Drag to rotate and explore connections 
                between identities, devices, and sessions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
