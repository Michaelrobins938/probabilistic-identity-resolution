import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useDemoStore } from '@/store/demoStore';
import { Card } from '@/components/ui/Card';

export const ClusteringViz: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { people, assignments, sessions } = useDemoStore();

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    // Clear previous content
    svg.selectAll('*').remove();

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Define scales
    const xScale = d3.scaleLinear()
      .domain([0, 24])
      .range([0, width - margin.left - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, 10])
      .range([height - margin.top - margin.bottom, 0]);

    // Draw person clusters (ellipses)
    people.forEach((person, index) => {
      const centerX = xScale(person.typicalHours.reduce((a, b) => a + b, 0) / person.typicalHours.length);
      const centerY = yScale(index * 2 + 2);
      
      // Cluster ellipse
      g.append('ellipse')
        .attr('cx', centerX)
        .attr('cy', centerY)
        .attr('rx', 80)
        .attr('ry', 60)
        .attr('fill', person.color)
        .attr('opacity', 0.15)
        .attr('stroke', person.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

      // Person label
      g.append('text')
        .attr('x', centerX)
        .attr('y', centerY)
        .attr('text-anchor', 'middle')
        .attr('fill', person.color)
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text(person.name.split(' ')[0]);

      // Stats label
      const personAssignments = assignments.filter(a => a.personId === person.id);
      if (personAssignments.length > 0) {
        g.append('text')
          .attr('x', centerX)
          .attr('y', centerY + 20)
          .attr('text-anchor', 'middle')
          .attr('fill', person.color)
          .attr('font-size', '11px')
          .text(`${personAssignments.length} sessions`);
      }
    });

    // Draw axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(12)
      .tickFormat(d => `${d}:00`);

    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(xAxis)
      .selectAll('text')
      .attr('font-size', '10px');

    // X axis label
    g.append('text')
      .attr('x', (width - margin.left - margin.right) / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#666')
      .text('Hour of Day');

    // Animate recent sessions
    const recentSessions = sessions.slice(0, 10);
    
    recentSessions.forEach((session, i) => {
      const person = people.find(p => p.id === 'person_a'); // Simplified - would use actual assignment
      if (!person) return;

      const targetX = xScale(session.hour);
      const targetY = yScale(people.findIndex(p => p.genres.includes(session.genre)) * 2 + 2);
      
      const startX = Math.random() * (width - margin.left - margin.right);
      const startY = Math.random() * (height - margin.top - margin.bottom);

      // Session point
      const circle = g.append('circle')
        .attr('cx', startX)
        .attr('cy', startY)
        .attr('r', 5)
        .attr('fill', person.color)
        .attr('opacity', 0.8);

      // Animate to cluster
      circle.transition()
        .duration(1000 + i * 100)
        .ease(d3.easeCubicOut)
        .attr('cx', targetX + (Math.random() - 0.5) * 40)
        .attr('cy', targetY + (Math.random() - 0.5) * 30)
        .attr('opacity', 0.4)
        .attr('r', 3);
    });

  }, [people, assignments, sessions]);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-900">Behavioral Clustering</h2>
        <span className="text-sm text-gray-500">
          {assignments.length} sessions clustered
        </span>
      </div>
      
      <svg 
        ref={svgRef} 
        width="100%" 
        height="400"
        viewBox="0 0 600 400"
        className="w-full"
      />
      
      <p className="mt-4 text-sm text-gray-600 text-center">
        Sessions are clustered based on time-of-day and content preferences. 
        Each ellipse represents a person&apos;s typical behavioral pattern.
      </p>
    </Card>
  );
};
