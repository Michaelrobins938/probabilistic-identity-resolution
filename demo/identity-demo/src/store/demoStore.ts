import { create } from 'zustand';
import { Person, Session, Assignment, DemoState } from '@/types';

const DEFAULT_PEOPLE: Person[] = [
  {
    id: 'person_a',
    name: 'Parent (Primary)',
    color: '#4F46E5',
    ageGroup: 'adult',
    devices: ['TV', 'Desktop'],
    genres: ['Drama', 'Documentary', 'Thriller'],
    timeOfDay: 'Evening 8-11 PM',
    typicalHours: [20, 21, 22, 23],
    avatar: '👤'
  },
  {
    id: 'person_b',
    name: 'Teenager',
    color: '#10B981',
    ageGroup: 'teen',
    devices: ['Mobile', 'Tablet'],
    genres: ['Sci-Fi', 'Action', 'Animation'],
    timeOfDay: 'Afternoon & Late Night',
    typicalHours: [15, 16, 20, 21, 22, 23],
    avatar: '🧑‍🎓'
  },
  {
    id: 'person_c',
    name: 'Child',
    color: '#F59E0B',
    ageGroup: 'child',
    devices: ['Tablet'],
    genres: ['Animation', 'Kids', 'Family'],
    timeOfDay: 'After School 4-6 PM',
    typicalHours: [16, 17, 18],
    avatar: '🧒'
  }
];

const CONTENT_LIBRARY = [
  { title: 'Stranger Things S4', genre: 'Sci-Fi', duration: 3600 },
  { title: 'Breaking Bad S1', genre: 'Drama', duration: 3600 },
  { title: 'Peppa Pig', genre: 'Kids', duration: 600 },
  { title: 'The Crown S5', genre: 'Drama', duration: 3600 },
  { title: 'Spider-Man: NWH', genre: 'Action', duration: 7200 },
  { title: 'Bluey', genre: 'Animation', duration: 420 },
  { title: 'The Office', genre: 'Comedy', duration: 1800 },
  { title: 'Planet Earth II', genre: 'Documentary', duration: 3000 },
  { title: 'Squid Game', genre: 'Thriller', duration: 3600 },
  { title: 'Avatar 2', genre: 'Sci-Fi', duration: 7200 }
];

const DEVICES = ['TV', 'Mobile', 'Tablet', 'Desktop'];

interface DemoStore extends DemoState {
  startSimulation: () => void;
  stopSimulation: () => void;
  addSession: () => void;
  clearSessions: () => void;
  setSpeed: (speed: number) => void;
  selectPerson: (personId: string | null) => void;
}

export const useDemoStore = create<DemoStore>((set, get) => ({
  people: DEFAULT_PEOPLE,
  sessions: [],
  assignments: [],
  isRunning: false,
  speed: 1,
  selectedPerson: null,
  totalSessions: 0,
  accuracy: 0.81,

  startSimulation: () => {
    set({ isRunning: true });
    
    const interval = setInterval(() => {
      if (!get().isRunning) {
        clearInterval(interval);
        return;
      }
      get().addSession();
    }, 2000 / get().speed);
  },

  stopSimulation: () => set({ isRunning: false }),

  addSession: () => {
    const state = get();
    const { people, sessions, assignments, totalSessions } = state;
    
    // Generate realistic session based on time
    const now = new Date();
    const hour = now.getHours();
    const dayOfWeek = now.getDay();
    
    // Pick most likely person based on hour
    let likelyPerson = people[0];
    let maxAffinity = 0;
    
    for (const person of people) {
      const hourMatch = person.typicalHours.includes(hour);
      const affinity = hourMatch ? 0.7 : 0.1;
      if (affinity > maxAffinity) {
        maxAffinity = affinity;
        likelyPerson = person;
      }
    }
    
    // Pick content matching person's genres
    const suitableContent = CONTENT_LIBRARY.filter(c => 
      likelyPerson.genres.includes(c.genre)
    );
    const content = suitableContent.length > 0 
      ? suitableContent[Math.floor(Math.random() * suitableContent.length)]
      : CONTENT_LIBRARY[Math.floor(Math.random() * CONTENT_LIBRARY.length)];
    
    // Pick device
    const device = likelyPerson.devices[
      Math.floor(Math.random() * likelyPerson.devices.length)
    ];
    
    const session: Session = {
      id: `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: now,
      device,
      content: content.title,
      genre: content.genre,
      duration: content.duration + (Math.random() - 0.5) * 300,
      hour,
      dayOfWeek
    };
    
    // Calculate probabilistic assignment
    const probabilities: Record<string, number> = {};
    let totalProb = 0;
    
    for (const person of people) {
      const hourMatch = person.typicalHours.includes(hour);
      const genreMatch = person.genres.includes(content.genre);
      const deviceMatch = person.devices.includes(device);
      
      let prob = 0.1; // Base probability
      if (hourMatch) prob += 0.4;
      if (genreMatch) prob += 0.3;
      if (deviceMatch) prob += 0.2;
      
      // Add some noise
      prob += (Math.random() - 0.5) * 0.1;
      prob = Math.max(0.05, prob);
      
      probabilities[person.id] = prob;
      totalProb += prob;
    }
    
    // Normalize
    for (const key in probabilities) {
      probabilities[key] /= totalProb;
    }
    
    // Select assigned person (with 80% accuracy to ground truth)
    const isCorrect = Math.random() < 0.8;
    const assignedPersonId = isCorrect 
      ? likelyPerson.id 
      : people[Math.floor(Math.random() * people.length)].id;
    
    const assignment: Assignment = {
      sessionId: session.id,
      personId: assignedPersonId,
      confidence: probabilities[assignedPersonId],
      probabilities,
      timestamp: now
    };
    
    set({
      sessions: [session, ...sessions].slice(0, 50),
      assignments: [assignment, ...assignments].slice(0, 50),
      totalSessions: totalSessions + 1
    });
  },

  clearSessions: () => set({ 
    sessions: [], 
    assignments: [], 
    totalSessions: 0 
  }),

  setSpeed: (speed) => set({ speed }),
  
  selectPerson: (personId) => set({ selectedPerson: personId })
}));
