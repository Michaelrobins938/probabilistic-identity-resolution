export interface Person {
  id: string;
  name: string;
  color: string;
  ageGroup: 'child' | 'teen' | 'adult' | 'senior';
  devices: string[];
  genres: string[];
  timeOfDay: string;
  typicalHours: number[];
  avatar: string;
}

export interface Session {
  id: string;
  timestamp: Date;
  device: string;
  content: string;
  genre: string;
  duration: number;
  hour: number;
  dayOfWeek: number;
}

export interface Assignment {
  sessionId: string;
  personId: string;
  confidence: number;
  probabilities: Record<string, number>;
  timestamp: Date;
}

export interface AttributionData {
  channel: string;
  accountLevelShare: number;
  personLevelShares: Record<string, number>;
  conversions: number;
  revenue: number;
}

export interface Cluster {
  id: string;
  x: number;
  y: number;
  radius: number;
  color: string;
  personId: string;
}

export interface DemoState {
  people: Person[];
  sessions: Session[];
  assignments: Assignment[];
  isRunning: boolean;
  speed: number;
  selectedPerson: string | null;
  totalSessions: number;
  accuracy: number;
}
