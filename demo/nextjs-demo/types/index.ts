export interface Person {
  id: string
  name: string
  avatar: string
  devices: Device[]
  color: string
}

export interface Device {
  id: string
  type: 'desktop' | 'mobile' | 'tablet'
  name: string
  icon: string
}

export interface Session {
  id: string
  personId: string
  deviceId: string
  timestamp: number
  page: string
  duration: number
  referrer: string
  converted: boolean
  revenue?: number
}

export interface Touchpoint {
  session: Session
  person: Person
  device: Device
  attributionWeight: number
}

export interface AttributionModel {
  name: string
  description: string
  calculate: (touchpoints: Touchpoint[]) => AttributionResult[]
}

export interface AttributionResult {
  personId: string
  deviceId: string
  contribution: number
  revenue: number
}

export interface Cluster {
  id: string
  personIds: string[]
  confidence: number
  center: { x: number; y: number }
}
