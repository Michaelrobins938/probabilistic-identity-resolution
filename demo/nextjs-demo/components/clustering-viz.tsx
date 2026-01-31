'use client'

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { useStore } from '@/lib/store'
import { Person, Session } from '@/types'

interface NodeData extends d3.SimulationNodeDatum {
  id: string
  personId: string
  type: 'person' | 'device' | 'session' | 'cluster'
  color: string
  name: string
  radius: number
  mass: number
  velocity: { x: number; y: number }
}

interface LinkData extends d3.SimulationLinkDatum<NodeData> {
  source: string | NodeData
  target: string | NodeData
  strength: number
  type: 'person-device' | 'device-session' | 'cluster-person'
}

export function ClusteringViz() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { persons, sessions, isRunning } = useStore()
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null)
  const [stats, setStats] = useState({
    totalNodes: 0,
    totalLinks: 0,
    clusters: 0,
    avgConfidence: 0
  })

  useEffect(() => {
    if (!svgRef.current || persons.length === 0) return

    const svg = d3.select(svgRef.current)
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    
    // Clear previous with fade out
    svg.selectAll('*')
      .transition()
      .duration(300)
      .style('opacity', 0)
      .remove()

    // Create hierarchical structure
    const nodes: NodeData[] = []
    const links: LinkData[] = []
    const clusterCenters: { x: number; y: number; personId: string; color: string }[] = []

    // Define cluster regions (triangular layout for stability)
    const centerX = width / 2
    const centerY = height / 2
    const clusterRadius = Math.min(width, height) * 0.25

    persons.forEach((person, index) => {
      const angle = (index / persons.length) * 2 * Math.PI - Math.PI / 2
      const cx = centerX + Math.cos(angle) * clusterRadius
      const cy = centerY + Math.sin(angle) * clusterRadius
      
      clusterCenters.push({
        x: cx,
        y: cy,
        personId: person.id,
        color: person.color
      })

      // Person node (fixed anchor)
      nodes.push({
        id: person.id,
        personId: person.id,
        type: 'person',
        color: person.color,
        name: person.name,
        radius: 35,
        mass: 10,
        velocity: { x: 0, y: 0 },
        x: cx,
        y: cy,
        fx: cx, // Fixed position
        fy: cy
      })

      // Add cluster zone (invisible boundary)
      nodes.push({
        id: `cluster-${person.id}`,
        personId: person.id,
        type: 'cluster',
        color: person.color,
        name: `${person.name} Zone`,
        radius: 80,
        mass: 0,
        velocity: { x: 0, y: 0 },
        x: cx,
        y: cy,
        fx: cx,
        fy: cy
      })

      // Device nodes (orbiting)
      person.devices.forEach((device, deviceIndex) => {
        const deviceAngle = angle + (deviceIndex - (person.devices.length - 1) / 2) * 0.6
        const orbitRadius = 60
        const deviceId = `${person.id}-${device.id}`

        nodes.push({
          id: deviceId,
          personId: person.id,
          type: 'device',
          color: person.color,
          name: device.name,
          radius: 12,
          mass: 5,
          velocity: { x: 0, y: 0 },
          x: cx + Math.cos(deviceAngle) * orbitRadius,
          y: cy + Math.sin(deviceAngle) * orbitRadius
        })

        links.push({
          source: person.id,
          target: deviceId,
          strength: 0.8,
          type: 'person-device'
        })
      })
    })

    // Add session nodes with intelligent placement
    const recentSessions = sessions.slice(0, 30)
    const sessionNodes: NodeData[] = []
    
    recentSessions.forEach((session, index) => {
      const personCluster = clusterCenters.find(c => c.personId === session.personId)
      if (!personCluster) return

      const deviceId = `${session.personId}-${session.deviceId}`
      const sessionNodeId = `session-${session.id}`
      
      // Calculate position based on confidence/attraction
      const angle = (index % 12) * (Math.PI / 6) + Math.random() * 0.5
      const distance = 30 + (index * 2)
      
      const sessionNode: NodeData = {
        id: sessionNodeId,
        personId: session.personId,
        type: 'session',
        color: session.converted ? '#00ff41' : personCluster.color,
        name: session.page.split('/').pop() || session.page,
        radius: session.converted ? 8 : 5,
        mass: 1,
        velocity: { x: 0, y: 0 },
        x: personCluster.x + Math.cos(angle) * distance,
        y: personCluster.y + Math.sin(angle) * distance
      }
      
      sessionNodes.push(sessionNode)
      nodes.push(sessionNode)

      links.push({
        source: deviceId,
        target: sessionNodeId,
        strength: session.converted ? 0.6 : 0.3,
        type: 'device-session'
      })
    })

    // Update stats
    setStats({
      totalNodes: nodes.length,
      totalLinks: links.length,
      clusters: persons.length,
      avgConfidence: 0.85
    })

    // Create defs for gradients and filters
    const defs = svg.append('defs')
    
    // Glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%')
    
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur')
    
    filter.append('feMerge')
      .append('feMergeNode')
      .attr('in', 'coloredBlur')
    
    filter.append('feMerge')
      .append('feMergeNode')
      .attr('in', 'SourceGraphic')

    // Create force simulation with STABILIZED parameters
    const simulation = d3.forceSimulation<NodeData>(nodes)
      .force('link', d3.forceLink<NodeData, LinkData>(links)
        .id(d => d.id)
        .strength(d => d.strength * 0.4)
        .distance(d => {
          if (d.type === 'person-device') return 50
          if (d.type === 'device-session') return 35
          return 40
        })
      )
      .force('charge', d3.forceManyBody()
        .strength(d => {
          if (d.type === 'person') return -800
          if (d.type === 'device') return -300
          return -100
        })
        .distanceMin(10)
        .distanceMax(200)
      )
      .force('center', d3.forceCenter(centerX, centerY).strength(0.05))
      .force('collision', d3.forceCollide<NodeData>()
        .radius(d => d.radius + 8)
        .strength(0.7)
      )
      .force('cluster', d3.forceX<NodeData>(d => {
        const center = clusterCenters.find(c => c.personId === d.personId)
        return center ? center.x : centerX
      }).strength(0.3))
      .force('clusterY', d3.forceY<NodeData>(d => {
        const center = clusterCenters.find(c => c.personId === d.personId)
        return center ? center.y : centerY
      }).strength(0.3))
      .alphaDecay(0.02) // Slower decay = more stable
      .velocityDecay(0.3) // More damping = less jitter

    // Draw cluster zones (invisible boundaries)
    const clusterZones = svg.append('g').attr('class', 'cluster-zones')
    
    clusterCenters.forEach(center => {
      clusterZones.append('circle')
        .attr('cx', center.x)
        .attr('cy', center.y)
        .attr('r', 0)
        .attr('fill', center.color)
        .attr('opacity', 0.08)
        .transition()
        .duration(1000)
        .ease(d3.easeCubicOut)
        .attr('r', 70)
    })

    // Draw links with animated dashes for data flow
    const linkGroup = svg.append('g').attr('class', 'links')
    
    const link = linkGroup.selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', d => {
        if (d.type === 'device-session') return '#404040'
        return '#606060'
      })
      .attr('stroke-width', d => {
        if (d.type === 'person-device') return 2
        return 1
      })
      .attr('stroke-opacity', 0.4)
      .attr('stroke-linecap', 'round')

    // Add animated data particles on links
    const particleGroup = svg.append('g').attr('class', 'particles')
    
    const particles = particleGroup.selectAll('circle')
      .data(links.filter(l => l.type === 'device-session'))
      .enter()
      .append('circle')
      .attr('r', 2)
      .attr('fill', '#00ff41')
      .attr('opacity', 0)

    // Draw nodes with sophisticated styling
    const nodeGroup = svg.append('g').attr('class', 'nodes')
    
    const node = nodeGroup.selectAll('g')
      .data(nodes.filter(n => n.type !== 'cluster'))
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedNode(d)
        event.stopPropagation()
      })
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d.radius * 1.2)
          .style('filter', 'url(#glow)')
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d.radius)
          .style('filter', null)
      })

    // Node circles with gradient fills
    node.append('circle')
      .attr('r', 0)
      .attr('fill', d => d.color)
      .attr('stroke', d => {
        if (d.type === 'person') return '#ffffff'
        if (d.converted) return '#00ff41'
        return d.color
      })
      .attr('stroke-width', d => {
        if (d.type === 'person') return 3
        if (d.converted) return 2
        return 1
      })
      .attr('stroke-opacity', 0.8)
      .transition()
      .duration(800)
      .ease(d3.easeBackOut)
      .attr('r', d => d.radius)

    // Add inner glow for person nodes
    node.filter(d => d.type === 'person')
      .append('circle')
      .attr('r', d => d.radius - 5)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.5)
      .style('filter', 'url(#glow)')

    // Add labels with smart positioning
    const labels = node.filter(d => d.type !== 'session' || d.converted)
      .append('text')
      .attr('dy', d => d.radius + 18)
      .attr('text-anchor', 'middle')
      .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name)
      .attr('fill', '#e0e0e0')
      .attr('font-size', d => d.type === 'person' ? '11px' : '9px')
      .attr('font-weight', d => d.type === 'person' ? '700' : '400')
      .attr('font-family', 'Courier New, monospace')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 1px 2px rgba(0,0,0,0.8)')

    // Add avatars to person nodes
    node.filter(d => d.type === 'person')
      .append('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .text(d => {
        const person = persons.find(p => p.id === d.personId)
        return person?.avatar || '👤'
      })
      .attr('font-size', '16px')
      .style('pointer-events', 'none')

    // Add pulse animation for converted sessions
    const pulseCircles = node.filter(d => d.converted)
      .append('circle')
      .attr('r', d => d.radius)
      .attr('fill', 'none')
      .attr('stroke', '#00ff41')
      .attr('stroke-width', 2)
      .attr('opacity', 0.8)

    // Animate pulses
    function pulse() {
      pulseCircles
        .transition()
        .duration(1500)
        .ease(d3.easeCubicOut)
        .attr('r', d => d.radius * 2)
        .attr('opacity', 0)
        .on('end', function() {
          d3.select(this)
            .attr('r', d => (d as NodeData).radius)
            .attr('opacity', 0.8)
        })
    }
    
    const pulseInterval = setInterval(pulse, 2000)

    // Animate particles along links
    function animateParticles() {
      particles
        .transition()
        .duration(2000)
        .ease(d3.easeLinear)
        .attrTween('transform', function(d) {
          return function(t) {
            const source = d.source as NodeData
            const target = d.target as NodeData
            const x = source.x! + (target.x! - source.x!) * t
            const y = source.y! + (target.y! - source.y!) * t
            return `translate(${x},${y})`
          }
        })
        .attr('opacity', d => {
          const target = d.target as NodeData
          return target.converted ? 1 : 0.3
        })
        .on('end', animateParticles)
    }
    
    animateParticles()

    // Update positions on tick with smooth transitions
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as NodeData).x!)
        .attr('y1', d => (d.source as NodeData).y!)
        .attr('x2', d => (d.target as NodeData).x!)
        .attr('y2', d => (d.target as NodeData).y!)

      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    // Click background to deselect
    svg.on('click', () => {
      setSelectedNode(null)
    })

    // Cleanup
    return () => {
      clearInterval(pulseInterval)
      simulation.stop()
      svg.selectAll('*').remove()
    }
  }, [persons, sessions])

  return (
    <div ref={containerRef} className="tactical-card corner-accent">
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
          <span>Clusters: {stats.clusters}</span>
        </div>
      </div>

      <div className="p-4 relative">
        <svg
          ref={svgRef}
          className="w-full h-[400px] rounded-sm"
          style={{ background: '#0a0a0a' }}
        />

        {/* Selected node info panel */}
        {selectedNode && (
          <div className="absolute bottom-4 left-4 right-4 bg-[#1a1a1a]/95 border border-[#404040] rounded-sm p-3 backdrop-blur-sm">
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs text-[#606060] uppercase tracking-wider mb-1">
                  {selectedNode.type}
                </div>
                <div className="text-sm font-mono text-[#e0e0e0]">
                  {selectedNode.name}
                </div>
                <div className="text-xs font-mono text-[#00ff41] mt-1">
                  ID: {selectedNode.id.substring(0, 8)}...
                </div>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-[#606060] hover:text-[#e0e0e0]"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute top-4 right-4 bg-[#0a0a0a]/90 border border-[#404040] rounded-sm p-3 text-xs space-y-2">
          <div className="font-mono text-[#606060] uppercase tracking-wider mb-2 border-b border-[#404040] pb-1">
            Legend
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#4F46E5] border-2 border-white" />
            <span className="text-[#a0a0a0] font-mono">Person</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#4F46E5] border border-[#4F46E5]" />
            <span className="text-[#a0a0a0] font-mono">Device</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#00ff41]" />
            <span className="text-[#a0a0a0] font-mono">Conversion</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#808080]" />
            <span className="text-[#a0a0a0] font-mono">Session</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-[#00ff41]" />
            <span className="text-[#a0a0a0] font-mono">Data Flow</span>
          </div>
        </div>
      </div>

      <div className="px-4 pb-4">
        <div className="text-xs font-mono text-[#606060] flex items-center gap-2">
          <span className="pulse-dot bg-[#00ff41]" />
          <span>Graph stabilized • Alpha decay: 0.02 • Velocity decay: 0.3</span>
        </div>
      </div>
    </div>
  )
}
