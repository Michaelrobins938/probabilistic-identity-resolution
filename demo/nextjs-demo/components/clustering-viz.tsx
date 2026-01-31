'use client'

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useStore } from '@/lib/store'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Person, Session } from '@/types'

interface NodeData extends d3.SimulationNodeDatum {
  id: string
  personId: string
  type: 'person' | 'device' | 'session'
  color: string
  name: string
  radius: number
}

interface LinkData extends d3.SimulationLinkDatum<NodeData> {
  source: string | NodeData
  target: string | NodeData
  strength: number
}

export function ClusteringViz() {
  const svgRef = useRef<SVGSVGElement>(null)
  const { persons, sessions, clusters, updateClusters } = useStore()

  useEffect(() => {
    if (!svgRef.current || persons.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    // Create nodes
    const nodes: NodeData[] = []
    const links: LinkData[] = []

    // Add person nodes
    persons.forEach((person, index) => {
      const angle = (index / persons.length) * 2 * Math.PI
      const radius = 150
      
      nodes.push({
        id: person.id,
        personId: person.id,
        type: 'person',
        color: person.color,
        name: person.name,
        radius: 30,
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        fx: width / 2 + Math.cos(angle) * radius,
        fy: height / 2 + Math.sin(angle) * radius,
      })

      // Add device nodes
      person.devices.forEach((device, deviceIndex) => {
        const deviceId = `${person.id}-${device.id}`
        const deviceAngle = angle + (deviceIndex - 0.5) * 0.5
        const deviceRadius = 80

        nodes.push({
          id: deviceId,
          personId: person.id,
          type: 'device',
          color: person.color,
          name: device.name,
          radius: 15,
          x: width / 2 + Math.cos(angle) * radius + Math.cos(deviceAngle) * deviceRadius,
          y: height / 2 + Math.sin(angle) * radius + Math.sin(deviceAngle) * deviceRadius,
        })

        links.push({
          source: person.id,
          target: deviceId,
          strength: 1,
        })
      })
    })

    // Add session nodes for recent sessions
    const recentSessions = sessions.slice(0, 20)
    recentSessions.forEach((session) => {
      const deviceId = `${session.personId}-${session.deviceId}`
      const sessionNodeId = `session-${session.id}`

      nodes.push({
        id: sessionNodeId,
        personId: session.personId,
        type: 'session',
        color: session.converted ? '#00D4AA' : '#808080',
        name: session.page,
        radius: session.converted ? 10 : 6,
      })

      links.push({
        source: deviceId,
        target: sessionNodeId,
        strength: session.converted ? 0.8 : 0.4,
      })
    })

    // Create force simulation
    const simulation = d3.forceSimulation<NodeData>(nodes)
      .force('link', d3.forceLink<NodeData, LinkData>(links)
        .id(d => d.id)
        .strength(d => d.strength * 0.5)
      )
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => d.radius + 5))

    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#333')
      .attr('stroke-width', d => Math.sqrt(d.strength * 2))
      .attr('stroke-opacity', 0.6)

    // Draw nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, NodeData>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          if (d.type !== 'person') {
            d.fx = null
            d.fy = null
          }
        })
      )

    // Add circles to nodes
    node.append('circle')
      .attr('r', d => d.radius)
      .attr('fill', d => d.color)
      .attr('opacity', d => d.type === 'session' ? 0.8 : 1)
      .attr('stroke', d => d.type === 'person' ? '#fff' : 'none')
      .attr('stroke-width', d => d.type === 'person' ? 2 : 0)

    // Add labels to person and device nodes
    node.filter(d => d.type !== 'session')
      .append('text')
      .attr('dy', d => d.radius + 15)
      .attr('text-anchor', 'middle')
      .text(d => d.name)
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('font-weight', '500')

    // Add avatars to person nodes
    node.filter(d => d.type === 'person')
      .append('text')
      .attr('dy', 4)
      .attr('text-anchor', 'middle')
      .text(d => {
        const person = persons.find(p => p.id === d.personId)
        return person?.avatar || ''
      })
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as NodeData).x!)
        .attr('y1', d => (d.source as NodeData).y!)
        .attr('x2', d => (d.target as NodeData).x!)
        .attr('y2', d => (d.target as NodeData).y!)

      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    return () => {
      simulation.stop()
    }
  }, [persons, sessions, clusters, updateClusters])

  return (
    <Card className="bg-netflix-black border-netflix-dark">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-semibold text-white">
          Identity Graph Visualization
        </CardTitle>
      </CardHeader>
      <CardContent>
        <svg
          ref={svgRef}
          className="w-full h-[350px]"
          style={{ background: '#0a0a0a' }}
        />
        <div className="mt-4 flex gap-4 justify-center text-xs text-netflix-gray">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-netflix-red" />
            <span>Person</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-netflix-gray" />
            <span>Device</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span>Conversion</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-gray-500" />
            <span>Session</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
