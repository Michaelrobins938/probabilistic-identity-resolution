'use client'

import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { useStore } from '@/lib/store'
import { Person, Session } from '@/types'

interface NodeData {
  id: string
  personId: string
  type: 'person' | 'device' | 'session'
  color: string
  name: string
  radius: number
  x: number
  y: number
  fx?: number | null
  fy?: number | null
}

interface LinkData {
  source: string
  target: string
  strength: number
  type: 'person-device' | 'device-session'
}

// Pre-calculate stable positions - NO physics after initial render
function calculateStableLayout(
  persons: Person[],
  sessions: Session[],
  width: number,
  height: number
): { nodes: NodeData[]; links: LinkData[] } {
  const nodes: NodeData[] = []
  const links: LinkData[] = []
  
  const centerX = width / 2
  const centerY = height / 2
  const clusterRadius = Math.min(width, height) * 0.28

  // Fixed positions for person nodes (triangular layout)
  persons.forEach((person, index) => {
    const angle = (index / persons.length) * 2 * Math.PI - Math.PI / 2
    const px = centerX + Math.cos(angle) * clusterRadius
    const py = centerY + Math.sin(angle) * clusterRadius
    
    nodes.push({
      id: person.id,
      personId: person.id,
      type: 'person',
      color: person.color,
      name: person.name,
      radius: 32,
      x: px,
      y: py,
      fx: px,
      fy: py
    })

    // Fixed positions for devices (orbital arrangement)
    const deviceRadius = 55
    person.devices.forEach((device, deviceIndex) => {
      const deviceAngle = angle + (deviceIndex - (person.devices.length - 1) / 2) * 0.5
      const dx = px + Math.cos(deviceAngle) * deviceRadius
      const dy = py + Math.sin(deviceAngle) * deviceRadius
      const deviceId = `${person.id}-${device.id}`

      nodes.push({
        id: deviceId,
        personId: person.id,
        type: 'device',
        color: person.color,
        name: device.name,
        radius: 12,
        x: dx,
        y: dy,
        fx: dx,
        fy: dy
      })

      links.push({
        source: person.id,
        target: deviceId,
        strength: 1,
        type: 'person-device'
      })
    })
  })

  // Calculate session positions with collision avoidance
  const recentSessions = sessions.slice(0, 25)
  const occupiedPositions: { x: number; y: number; radius: number }[] = []

  recentSessions.forEach((session, index) => {
    const person = persons.find(p => p.id === session.personId)
    if (!person) return

    const personNode = nodes.find(n => n.id === person.id)
    if (!personNode) return

    const deviceId = `${session.personId}-${session.deviceId}`
    const deviceNode = nodes.find(n => n.id === deviceId)
    if (!deviceNode) return

    // Find position near device with spiral collision detection
    let angle = (index * 0.8) % (2 * Math.PI)
    let distance = 25 + (index * 1.5)
    let attempts = 0
    let sx = deviceNode.x + Math.cos(angle) * distance
    let sy = deviceNode.y + Math.sin(angle) * distance
    const radius = session.converted ? 8 : 5

    // Check collisions and adjust
    while (attempts < 20) {
      let collision = false
      for (const pos of occupiedPositions) {
        const dx = sx - pos.x
        const dy = sy - pos.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < (radius + pos.radius + 8)) {
          collision = true
          break
        }
      }
      
      if (!collision) break
      
      // Spiral outward
      angle += 0.5
      distance += 8
      sx = deviceNode.x + Math.cos(angle) * distance
      sy = deviceNode.y + Math.sin(angle) * distance
      attempts++
    }

    occupiedPositions.push({ x: sx, y: sy, radius })

    const sessionNodeId = `session-${session.id}`
    nodes.push({
      id: sessionNodeId,
      personId: session.personId,
      type: 'session',
      color: session.converted ? '#00ff41' : person.color,
      name: session.page.split('/').pop() || session.page,
      radius,
      x: sx,
      y: sy,
      fx: sx,
      fy: sy
    })

    links.push({
      source: deviceId,
      target: sessionNodeId,
      strength: session.converted ? 0.8 : 0.4,
      type: 'device-session'
    })
  })

  return { nodes, links }
}

export function ClusteringViz() {
  const svgRef = useRef<SVGSVGElement>(null)
  const { persons, sessions, isRunning } = useStore()
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 })
  
  // Calculate layout once - NO physics simulation
  const layout = useMemo(() => {
    if (persons.length === 0) return { nodes: [], links: [] }
    return calculateStableLayout(persons, sessions, dimensions.width, dimensions.height)
  }, [persons, sessions, dimensions])

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current?.parentElement) {
        const rect = svgRef.current.parentElement.getBoundingClientRect()
        setDimensions({
          width: rect.width,
          height: 400
        })
      }
    }
    
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Render static SVG - NO continuous updates
  useEffect(() => {
    if (!svgRef.current || layout.nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const { nodes, links } = layout

    // Create defs
    const defs = svg.append('defs')
    
    // Glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%')
    
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur')
    
    const feMerge = filter.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'coloredBlur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    // Background grid
    const gridGroup = svg.append('g').attr('class', 'grid')
    
    // Draw concentric zones around each person
    persons.forEach(person => {
      const personNode = nodes.find(n => n.id === person.id)
      if (!personNode) return
      
      // Outer zone
      gridGroup.append('circle')
        .attr('cx', personNode.x)
        .attr('cy', personNode.y)
        .attr('r', 90)
        .attr('fill', person.color)
        .attr('opacity', 0.03)
        .attr('stroke', person.color)
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.1)
        .attr('stroke-dasharray', '5,5')
      
      // Inner zone
      gridGroup.append('circle')
        .attr('cx', personNode.x)
        .attr('cy', personNode.y)
        .attr('r', 60)
        .attr('fill', person.color)
        .attr('opacity', 0.05)
    })

    // Draw links
    const linkGroup = svg.append('g').attr('class', 'links')
    
    linkGroup.selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('x1', d => {
        const source = nodes.find(n => n.id === d.source)
        return source?.x || 0
      })
      .attr('y1', d => {
        const source = nodes.find(n => n.id === d.source)
        return source?.y || 0
      })
      .attr('x2', d => {
        const target = nodes.find(n => n.id === d.target)
        return target?.x || 0
      })
      .attr('y2', d => {
        const target = nodes.find(n => n.id === d.target)
        return target?.y || 0
      })
      .attr('stroke', d => {
        if (d.type === 'device-session') return '#333'
        return '#555'
      })
      .attr('stroke-width', d => d.type === 'person-device' ? 2 : 1)
      .attr('stroke-opacity', 0.5)

    // Animated data particles (only animation - nodes are static)
    const particleGroup = svg.append('g').attr('class', 'particles')
    
    const sessionLinks = links.filter(l => l.type === 'device-session')
    
    sessionLinks.forEach((link, i) => {
      const source = nodes.find(n => n.id === link.source)
      const target = nodes.find(n => n.id === link.target)
      if (!source || !target) return

      const particle = particleGroup.append('circle')
        .attr('r', 2.5)
        .attr('fill', target.color === '#00ff41' ? '#00ff41' : '#888')
        .attr('opacity', 0)

      // Animate along path
      function animate() {
        particle
          .attr('opacity', 1)
          .transition()
          .duration(1500 + Math.random() * 1000)
          .ease(d3.easeLinear)
          .attrTween('transform', () => {
            return (t: number) => {
              const x = source.x + (target.x - source.x) * t
              const y = source.y + (target.y - source.y) * t
              return `translate(${x},${y})`
            }
          })
          .on('end', () => {
            particle.attr('opacity', 0)
            setTimeout(animate, Math.random() * 2000)
          })
      }
      
      setTimeout(animate, i * 100)
    })

    // Draw nodes - COMPLETELY STATIC
    const nodeGroup = svg.append('g').attr('class', 'nodes')
    
    const node = nodeGroup.selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .style('cursor', d => d.type === 'session' ? 'default' : 'pointer')
      .on('click', (event, d) => {
        if (d.type !== 'session') {
          setSelectedNode(d)
          event.stopPropagation()
        }
      })
      .on('mouseenter', (event, d) => {
        if (d.type !== 'session') {
          setHoveredNode(d.id)
        }
      })
      .on('mouseleave', () => {
        setHoveredNode(null)
      })

    // Person nodes with rings
    const personNodes = node.filter(d => d.type === 'person')
    
    // Outer ring
    personNodes.append('circle')
      .attr('r', d => d.radius + 4)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.3)
    
    // Main circle
    personNodes.append('circle')
      .attr('r', d => d.radius)
      .attr('fill', d => d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
    
    // Glow on hover
    personNodes.append('circle')
      .attr('class', 'hover-glow')
      .attr('r', d => d.radius + 8)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 3)
      .attr('opacity', 0)
      .style('filter', 'url(#glow)')
    
    // Avatar
    personNodes.append('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .text(d => {
        const person = persons.find(p => p.id === d.personId)
        return person?.avatar || '👤'
      })
      .attr('font-size', '18px')
      .style('pointer-events', 'none')
    
    // Label
    personNodes.append('text')
      .attr('dy', d => d.radius + 20)
      .attr('text-anchor', 'middle')
      .text(d => d.name)
      .attr('fill', '#e0e0e0')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('font-family', 'system-ui, sans-serif')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 1px 3px rgba(0,0,0,0.8)')

    // Device nodes
    const deviceNodes = node.filter(d => d.type === 'device')
    
    deviceNodes.append('circle')
      .attr('r', d => d.radius)
      .attr('fill', '#1a1a1a')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 2)
    
    deviceNodes.append('text')
      .attr('dy', d => d.radius + 14)
      .attr('text-anchor', 'middle')
      .text(d => d.name.length > 12 ? d.name.substring(0, 12) + '...' : d.name)
      .attr('fill', '#a0a0a0')
      .attr('font-size', '9px')
      .attr('font-family', 'system-ui, sans-serif')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 1px 2px rgba(0,0,0,0.8)')

    // Session nodes
    const sessionNodes = node.filter(d => d.type === 'session')
    
    sessionNodes.append('circle')
      .attr('r', d => d.radius)
      .attr('fill', d => d.color)
      .attr('opacity', 0.9)
    
    // Pulsing ring for conversions
    const conversionNodes = node.filter(d => d.type === 'session' && d.color === '#00ff41')
    
    conversionNodes.append('circle')
      .attr('class', 'pulse-ring')
      .attr('r', d => d.radius)
      .attr('fill', 'none')
      .attr('stroke', '#00ff41')
      .attr('stroke-width', 2)
      .attr('opacity', 0.8)

    // Update hover effects (CSS-based, no physics)
    node.selectAll('.hover-glow')
      .attr('opacity', d => hoveredNode === d.id ? 0.6 : 0)

  }, [layout, hoveredNode, persons])

  // Pulsing animation for conversions
  useEffect(() => {
    if (!svgRef.current) return
    
    const interval = setInterval(() => {
      d3.select(svgRef.current)
        .selectAll('.pulse-ring')
        .transition()
        .duration(1200)
        .ease(d3.easeCubicOut)
        .attr('r', 20)
        .attr('opacity', 0)
        .on('end', function() {
          d3.select(this)
            .attr('r', d => (d as NodeData).radius)
            .attr('opacity', 0.8)
        })
    }, 1500)
    
    return () => clearInterval(interval)
  }, [layout])

  const stats = useMemo(() => ({
    totalNodes: layout.nodes.length,
    totalLinks: layout.links.length,
    clusters: persons.length,
    sessions: layout.nodes.filter(n => n.type === 'session').length
  }), [layout, persons.length])

  return (
    <div className="tactical-card corner-accent">
      <div className="tactical-header">
        <div className="flex items-center gap-3">
          <span className="section-header !mb-0">Identity Graph Network</span>
          <div className={`status-indicator ${isRunning ? 'status-operational' : 'status-info'} text-xs`}>
            {isRunning ? '● LIVE' : '○ PAUSED'}
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono text-[#606060]">
          <span>Nodes: {stats.totalNodes}</span>
          <span>Links: {stats.totalLinks}</span>
          <span>Sessions: {stats.sessions}</span>
        </div>
      </div>

      <div className="p-4 relative" onClick={() => setSelectedNode(null)}>
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="w-full rounded-sm"
          style={{ background: '#0a0a0a', minHeight: '400px' }}
        />

        {/* Selected node info */}
        {selectedNode && (
          <div className="absolute bottom-4 left-4 right-4 bg-[#1a1a1a]/95 border border-[#404040] rounded-sm p-3 backdrop-blur-sm z-10">
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs text-[#606060] uppercase tracking-wider mb-1">
                  {selectedNode.type}
                </div>
                <div className="text-sm font-semibold text-[#e0e0e0]">
                  {selectedNode.name}
                </div>
                <div className="text-xs font-mono text-[#00ff41] mt-1">
                  Fixed Position: ({selectedNode.x.toFixed(0)}, {selectedNode.y.toFixed(0)})
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedNode(null)
                }}
                className="text-[#606060] hover:text-[#e0e0e0] px-2"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute top-4 right-4 bg-[#0a0a0a]/95 border border-[#404040] rounded-sm p-3 text-xs space-y-2">
          <div className="font-mono text-[#606060] uppercase tracking-wider mb-2 border-b border-[#404040] pb-1">
            Legend
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#4F46E5] border-2 border-white" />
            <span className="text-[#a0a0a0]">Person</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#1a1a1a] border-2 border-[#4F46E5]" />
            <span className="text-[#a0a0a0]">Device</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#00ff41]" />
            <span className="text-[#a0a0a0]">Conversion</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#666]" />
            <span className="text-[#a0a0a0]">Session</span>
          </div>
        </div>
      </div>

      <div className="px-4 pb-4">
        <div className="text-xs font-mono text-[#606060] flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#00ff41]" />
          <span>Static Layout • No Physics • Fixed Positions • Click nodes for details</span>
        </div>
      </div>
    </div>
  )
}
