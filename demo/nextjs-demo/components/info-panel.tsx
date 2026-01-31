'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ComponentInfo {
  id: string
  name: string
  category: 'pipeline' | 'ml' | 'ui' | 'data'
  description: string
  technicalSpecs: {
    technology: string
    latency: string
    throughput: string
    accuracy?: string
  }
  useCase: string
  codeExample: string
  learnMoreLinks: { title: string; url: string }[]
}

const componentLibrary: ComponentInfo[] = [
  {
    id: 'event-ingestion',
    name: 'Event Ingestion Pipeline',
    category: 'pipeline',
    description: 'High-throughput event collection system that captures touchpoints from websites, mobile apps, and external sources. Handles schema validation, deduplication, and initial enrichment.',
    technicalSpecs: {
      technology: 'Apache Kafka (24 partitions), JSON Schema',
      latency: '<10ms P99',
      throughput: '100K events/second',
    },
    useCase: 'Capture every interaction across all channels without data loss, even during traffic spikes.',
    codeExample: `// SDK Event Capture
ir.track({
  event: 'page_view',
  properties: {
    url: window.location.href,
    referrer: document.referrer,
    device_fingerprint: fp.get()
  },
  timestamp: Date.now()
});`,
    learnMoreLinks: [
      { title: 'View Whitepaper', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'System Architecture', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/STRESS_TEST_REPORT.md' },
    ],
  },
  {
    id: 'feature-extraction',
    name: 'Feature Extraction Engine',
    category: 'ml',
    description: 'Real-time feature engineering that extracts 47+ behavioral and technical features from each event. Uses stateful stream processing to compute temporal and sequential patterns.',
    technicalSpecs: {
      technology: 'Apache Flink, Stateful Stream Processing',
      latency: '<25ms P99',
      throughput: '50K events/second',
    },
    useCase: 'Transform raw events into ML-ready feature vectors that capture user behavior patterns across sessions.',
    codeExample: `# Feature Definition
features = [
  BrowserFingerprint(),      # Canvas, WebGL hashes
  TemporalPattern(),         # Time of day, day of week
  NavigationBehavior(),      # Scroll depth, click patterns
  DeviceCharacteristics(),   # Screen, hardware info
  NetworkSignature(),        # Connection type, latency
]`,
    learnMoreLinks: [
      { title: 'Feature Catalog', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'Flink Architecture', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/README.md' },
    ],
  },
  {
    id: 'clustering-algorithm',
    name: 'ML Clustering Algorithm',
    category: 'ml',
    description: 'Ensemble machine learning models that group similar digital fingerprints into identity clusters. Combines supervised and unsupervised learning for robust matching.',
    technicalSpecs: {
      technology: 'XGBoost + Isolation Forest, TensorFlow',
      latency: '<50ms inference',
      throughput: '25K inferences/second',
      accuracy: '94.7% AUC-ROC',
    },
    useCase: 'Resolve anonymous visitors to persistent identities across devices and sessions without relying on cookies.',
    codeExample: `# Model Inference
cluster = ml.predict({
  features: event.feature_vector,
  historical_clusters: cache.get(nearest_neighbors),
  temporal_context: session.window_metrics
})

confidence = calibrate(cluster.probability)`,
    learnMoreLinks: [
      { title: 'Model Architecture', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'Accuracy Metrics', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/STRESS_TEST_REPORT.md' },
    ],
  },
  {
    id: 'identity-graph',
    name: 'Identity Graph Database',
    category: 'data',
    description: 'Graph database storing resolved identities, device relationships, and historical touchpoints. Enables complex journey queries and relationship analysis.',
    technicalSpecs: {
      technology: 'Neo4j, GraphQL API',
      latency: '<100ms for 6-hop queries',
      throughput: '10K queries/second',
    },
    useCase: 'Query complete customer journeys across all touchpoints to understand path-to-conversion.',
    codeExample: `// Graph Query Example
MATCH path = (start:Event)-[:FOLLOWED_BY*1..10]->(conversion:Conversion)
WHERE start.device_id = $deviceId
RETURN 
  nodes(path) as touchpoints,
  relationships(path) as transitions,
  length(path) as journey_length`,
    learnMoreLinks: [
      { title: 'Graph Schema', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'Query Examples', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/README.md' },
    ],
  },
  {
    id: 'attribution-engine',
    name: 'Attribution Engine',
    category: 'ml',
    description: 'Multi-touch attribution system supporting multiple models: First-touch, Last-touch, Linear, Time-decay, and Data-driven (Shapley values).',
    technicalSpecs: {
      technology: 'Shapley Values, CUPED, Cooperative Game Theory',
      latency: '<200ms for complex journeys',
      throughput: '5K attributions/second',
    },
    useCase: 'Accurately distribute conversion credit across all touchpoints to optimize marketing spend.',
    codeExample: `# Attribution Calculation
attribution = Attribution.shapley_value(
  touchpoints=journey.events,
  conversion=journey.conversion,
  model='data_driven',
  lookback_days=30
)

# Returns contribution scores for each channel`,
    learnMoreLinks: [
      { title: 'Attribution Models', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'Shapley Values Explained', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/BUSINESS_CASE.md' },
    ],
  },
  {
    id: 'confidence-scoring',
    name: 'Confidence Scoring System',
    category: 'ml',
    description: 'Uncertainty quantification system that provides calibrated confidence scores for each identity resolution. Uses Monte Carlo dropout and ensemble disagreement.',
    technicalSpecs: {
      technology: 'Monte Carlo Dropout, Platt Scaling',
      latency: '<5ms',
      throughput: '50K scores/second',
      accuracy: 'Well-calibrated (ECE < 0.02)',
    },
    useCase: 'Make risk-weighted decisions based on match confidence. High-confidence matches for automation, low-confidence for review.',
    codeExample: `# Confidence Calculation
confidence = Confidence.score(
  prediction=cluster_assignment,
  model_uncertainty=mc_dropout_variance,
  feature_stability=temporal_consistency,
  historical_accuracy=validation_results
)

# Returns calibrated probability`,
    learnMoreLinks: [
      { title: 'Uncertainty Quantification', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
      { title: 'Calibration Methods', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/STRESS_TEST_REPORT.md' },
    ],
  },
  {
    id: 'session-feed',
    name: 'Real-Time Session Feed',
    category: 'ui',
    description: 'Live streaming interface showing active user sessions, page views, and conversion events. Updates in real-time via WebSocket connection.',
    technicalSpecs: {
      technology: 'WebSocket, Server-Sent Events',
      latency: '<100ms end-to-end',
      throughput: '10K concurrent sessions',
    },
    useCase: 'Monitor live user activity for real-time analytics and operational awareness.',
    codeExample: `// WebSocket Connection
const ws = new WebSocket('wss://api.example.com/sessions');

ws.onmessage = (event) => {
  const session = JSON.parse(event.data);
  displaySession(session);
};`,
    learnMoreLinks: [
      { title: 'WebSocket API', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/README.md' },
      { title: 'Event Streaming', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
    ],
  },
  {
    id: 'clustering-viz',
    name: 'Clustering Visualization',
    category: 'ui',
    description: 'Interactive 2D/3D visualization of identity clusters. Shows how ML groups similar entities and highlights edge cases.',
    technicalSpecs: {
      technology: 'D3.js, WebGL, t-SNE projection',
      latency: '60fps rendering',
      throughput: '1,000 nodes interactive',
    },
    useCase: 'Visualize and debug clustering results. Identify misclassifications and tune model parameters.',
    codeExample: `// t-SNE Projection for Viz
projection = TSNE.transform(
  data=feature_vectors,
  perplexity=30,
  iterations=1000
)

renderScatterPlot(projection, color_by='cluster_id')`,
    learnMoreLinks: [
      { title: 'Visualization API', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/README.md' },
      { title: 'Clustering Debug', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/STRESS_TEST_REPORT.md' },
    ],
  },
  {
    id: 'attribution-dashboard',
    name: 'Attribution Dashboard',
    category: 'ui',
    description: 'Comprehensive analytics dashboard showing attribution breakdowns, channel performance, and ROI metrics across all models.',
    technicalSpecs: {
      technology: 'React, D3.js, TimescaleDB',
      latency: '<500ms for complex queries',
      throughput: '100 concurrent users',
    },
    useCase: 'Analyze marketing performance with accurate multi-touch attribution instead of last-click bias.',
    codeExample: `// Dashboard Query
const metrics = await api.query({
  dateRange: [start, end],
  dimensions: ['channel', 'campaign'],
  metrics: ['attributed_revenue', 'touchpoints', 'roas'],
  attributionModel: 'shapley_value'
});`,
    learnMoreLinks: [
      { title: 'Dashboard API', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/README.md' },
      { title: 'Analytics Guide', url: 'https://github.com/Michaelrobins938/probabilistic-identity-resolution/blob/main/WHITEPAPER.md' },
    ],
  },
]

const categories = [
  { id: 'all', name: 'All Components', color: '#00ff41' },
  { id: 'pipeline', name: 'Data Pipeline', color: '#00bfff' },
  { id: 'ml', name: 'ML Systems', color: '#ffb800' },
  { id: 'data', name: 'Data Storage', color: '#ff6b35' },
  { id: 'ui', name: 'User Interface', color: '#e0e0e0' },
]

export function InfoPanel() {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedComponent, setSelectedComponent] = useState<ComponentInfo | null>(null)

  const filteredComponents = selectedCategory === 'all'
    ? componentLibrary
    : componentLibrary.filter(c => c.category === selectedCategory)

  const getCategoryColor = (category: string) => {
    return categories.find(c => c.id === category)?.color || '#e0e0e0'
  }

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed right-6 top-1/2 -translate-y-1/2 z-40 tactical-btn flex flex-col items-center gap-1 py-4 px-3 ${
          isOpen ? 'border-[#00ff41] text-[#00ff41]' : ''
        }`}
        style={{
          writingMode: isOpen ? 'horizontal-tb' : 'vertical-rl',
          textOrientation: 'mixed',
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4M12 8h.01" />
        </svg>
        <span className="text-xs font-bold tracking-wider">
          {isOpen ? 'Close Info Panel' : 'Component Info'}
        </span>
      </button>

      {/* Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-[450px] max-w-full tactical-card z-50 flex flex-col"
          >
            {/* Header */}
            <div className="tactical-header border-b-2 border-[#404040]">
              <div className="flex items-center gap-2">
                <span className="text-[#00ff41] text-lg">📚</span>
                <span className="text-sm font-bold tracking-wider uppercase text-[#e0e0e0]">
                  Component Reference
                </span>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-[#606060] hover:text-[#a0a0a0] text-xl"
              >
                ×
              </button>
            </div>

            {/* Category Filter */}
            <div className="p-4 border-b border-[#404040]">
              <div className="flex flex-wrap gap-2">
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => {
                      setSelectedCategory(cat.id)
                      setSelectedComponent(null)
                    }}
                    className={`px-3 py-1.5 rounded-sm text-xs font-bold uppercase tracking-wider transition-all ${
                      selectedCategory === cat.id
                        ? 'text-[#1a1a1a]'
                        : 'border border-[#404040] text-[#a0a0a0] hover:border-[#e0e0e0] hover:text-[#e0e0e0]'
                    }`}
                    style={{
                      backgroundColor: selectedCategory === cat.id ? cat.color : 'transparent',
                    }}
                  >
                    {cat.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Component List or Detail View */}
            <div className="flex-1 overflow-auto">
              <AnimatePresence mode="wait">
                {!selectedComponent ? (
                  <motion.div
                    key="list"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="p-4 space-y-2"
                  >
                    {filteredComponents.map((component) => (
                      <button
                        key={component.id}
                        onClick={() => setSelectedComponent(component)}
                        className="w-full text-left tactical-card p-3 hover:border-[#00ff41]/50 transition-all group"
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2">
                              <div
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: getCategoryColor(component.category) }}
                              />
                              <span className="font-bold text-[#e0e0e0] group-hover:text-[#00ff41] transition-colors">
                                {component.name}
                              </span>
                            </div>
                            <p className="text-xs text-[#606060] mt-1 line-clamp-2">
                              {component.description}
                            </p>
                          </div>
                          <span className="text-[#606060] group-hover:text-[#00ff41]">›</span>
                        </div>
                      </button>
                    ))}
                  </motion.div>
                ) : (
                  <motion.div
                    key="detail"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="p-4 space-y-4"
                  >
                    {/* Back button */}
                    <button
                      onClick={() => setSelectedComponent(null)}
                      className="text-xs font-mono text-[#606060] hover:text-[#a0a0a0] flex items-center gap-1"
                    >
                      ‹ Back to list
                    </button>

                    {/* Title */}
                    <div className="border-b border-[#404040] pb-4">
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: getCategoryColor(selectedComponent.category) }}
                        />
                        <span className="text-xs font-bold text-[#606060] uppercase tracking-wider">
                          {categories.find(c => c.id === selectedComponent.category)?.name}
                        </span>
                      </div>
                      <h2 className="text-lg font-bold text-[#e0e0e0]">{selectedComponent.name}</h2>
                    </div>

                    {/* Description */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold text-[#00ff41] uppercase tracking-wider">
                        Description
                      </div>
                      <p className="text-sm text-[#a0a0a0] leading-relaxed">
                        {selectedComponent.description}
                      </p>
                    </div>

                    {/* Technical Specs */}
                    <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-4 space-y-3">
                      <div className="text-xs font-bold text-[#00bfff] uppercase tracking-wider flex items-center gap-2">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                        </svg>
                        Technical Specifications
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        <div>
                          <span className="text-[#606060]">Technology:</span>
                          <p className="text-[#e0e0e0] mt-1">{selectedComponent.technicalSpecs.technology}</p>
                        </div>
                        <div>
                          <span className="text-[#606060]">Latency:</span>
                          <p className="text-[#e0e0e0] mt-1">{selectedComponent.technicalSpecs.latency}</p>
                        </div>
                        <div>
                          <span className="text-[#606060]">Throughput:</span>
                          <p className="text-[#e0e0e0] mt-1">{selectedComponent.technicalSpecs.throughput}</p>
                        </div>
                        {selectedComponent.technicalSpecs.accuracy && (
                          <div>
                            <span className="text-[#606060]">Accuracy:</span>
                            <p className="text-[#e0e0e0] mt-1">{selectedComponent.technicalSpecs.accuracy}</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Use Case */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold text-[#ffb800] uppercase tracking-wider">
                        Real-World Use Case
                      </div>
                      <p className="text-sm text-[#a0a0a0] leading-relaxed border-l-2 border-[#ffb800]/50 pl-3">
                        {selectedComponent.useCase}
                      </p>
                    </div>

                    {/* Code Example */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider">
                        Code Example
                      </div>
                      <div className="bg-[#0f0f0f] border border-[#404040] rounded-sm p-3 overflow-x-auto">
                        <pre className="text-xs font-mono text-[#e0e0e0] whitespace-pre">
                          <code>{selectedComponent.codeExample}</code>
                        </pre>
                      </div>
                    </div>

                    {/* Learn More Links */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold text-[#00ff41] uppercase tracking-wider">
                        Learn More
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {selectedComponent.learnMoreLinks.map((link, idx) => (
                          <a
                            key={idx}
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-3 py-1.5 border border-[#404040] rounded-sm text-xs text-[#00bfff] hover:border-[#00bfff] transition-all"
                          >
                            {link.title} ›
                          </a>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Footer */}
            <div className="tactical-header border-t-2 border-[#404040] text-center">
              <span className="text-[10px] font-mono text-[#606060]">
                {filteredComponents.length} components available
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
