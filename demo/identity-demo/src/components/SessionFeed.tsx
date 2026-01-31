import React from 'react';
import { useDemoStore } from '@/store/demoStore';
import { Card } from '@/components/ui/Card';
import { Smartphone, Monitor, Tv, Tablet, Play, Pause, RotateCcw, Zap } from 'lucide-react';

const deviceIcons: Record<string, React.ReactNode> = {
  'Mobile': <Smartphone className="w-4 h-4" />,
  'Desktop': <Monitor className="w-4 h-4" />,
  'TV': <Tv className="w-4 h-4" />,
  'Tablet': <Tablet className="w-4 h-4" />
};

const confidenceColor = (confidence: number): string => {
  if (confidence >= 0.7) return 'bg-green-500';
  if (confidence >= 0.5) return 'bg-yellow-500';
  return 'bg-orange-500';
};

export const SessionFeed: React.FC = () => {
  const { 
    sessions, 
    assignments, 
    people, 
    isRunning, 
    speed,
    startSimulation, 
    stopSimulation, 
    clearSessions,
    setSpeed,
    selectedPerson 
  } = useDemoStore();

  const getPersonById = (id: string) => people.find(p => p.id === id);

  const filteredAssignments = selectedPerson
    ? assignments.filter(a => a.personId === selectedPerson)
    : assignments;

  const filteredSessions = selectedPerson
    ? sessions.filter(s => filteredAssignments.some(a => a.sessionId === s.id))
    : sessions;

  return (
    <Card className="p-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Live Session Stream</h2>
          <p className="text-sm text-gray-500 mt-1">
            Real-time probabilistic identity resolution
            {selectedPerson && (
              <span className="ml-2 text-indigo-600 font-medium">
                (Filtered: {getPersonById(selectedPerson)?.name})
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Speed Control */}
          <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1 mr-2">
            <Zap className="w-4 h-4 text-gray-500" />
            {[1, 2, 5].map(s => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  speed === s 
                    ? 'bg-white text-indigo-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          {/* Start/Stop */}
          <button
            onClick={isRunning ? stopSimulation : startSimulation}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
              isRunning 
                ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isRunning ? (
              <><Pause className="w-4 h-4" /> Stop</>
            ) : (
              <><Play className="w-4 h-4" /> Start Simulation</>
            )}
          </button>

          <button
            onClick={clearSessions}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="Clear all sessions"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-50 rounded-xl p-4">
          <div className="text-2xl font-bold text-gray-900">{sessions.length}</div>
          <div className="text-xs text-gray-500">Active Sessions</div>
        </div>
        <div className="bg-gray-50 rounded-xl p-4">
          <div className="text-2xl font-bold text-gray-900">
            {(assignments.reduce((sum, a) => sum + a.confidence, 0) / (assignments.length || 1) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Avg Confidence</div>
        </div>
        <div className="bg-gray-50 rounded-xl p-4">
          <div className="text-2xl font-bold text-gray-900">
            {((assignments.filter(a => a.confidence >= 0.7).length / (assignments.length || 1)) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">High Confidence</div>
        </div>
      </div>

      {/* Sessions List */}
      <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
        {filteredSessions.slice(0, 20).map((session, index) => {
          const assignment = assignments.find(a => a.sessionId === session.id);
          const assignedPerson = assignment ? getPersonById(assignment.personId) : null;
          const isRecent = index < 3;

          return (
            <div 
              key={session.id}
              className={`
                flex items-center gap-4 p-4 rounded-xl border transition-all duration-500
                ${isRecent ? 'bg-gradient-to-r from-indigo-50 to-white border-indigo-200' : 'bg-white border-gray-200'}
              `}
            >
              {/* Device Icon */}
              <div className={`
                w-10 h-10 rounded-lg flex items-center justify-center
                ${assignedPerson ? '' : 'bg-gray-100 text-gray-400'}
              `}
              style={assignedPerson ? { 
                backgroundColor: `${assignedPerson.color}15`,
                color: assignedPerson.color 
              } : {}}
              >
                {deviceIcons[session.device] || <Tv className="w-4 h-4" />}
              </div>

              {/* Content Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-900 truncate">{session.content}</span>
                  <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full">
                    {session.genre}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                  <span>{session.device}</span>
                  <span>•</span>
                  <span>{session.hour}:00</span>
                  <span>•</span>
                  <span>{Math.round(session.duration / 60)} min</span>
                </div>
              </div>

              {/* Assignment */}
              {assignment && assignedPerson && (
                <div className="flex items-center gap-3">
                  {/* Probability Distribution */}
                  <div className="hidden sm:flex items-center gap-1">
                    {people.map(person => {
                      const prob = assignment.probabilities[person.id] || 0;
                      return (
                        <div 
                          key={person.id}
                          className="w-1.5 rounded-full transition-all duration-500"
                          style={{ 
                            height: `${Math.max(8, prob * 32)}px`,
                            backgroundColor: person.id === assignment.personId ? person.color : '#e5e7eb'
                          }}
                          title={`${person.name}: ${(prob * 100).toFixed(1)}%`}
                        />
                      );
                    })}
                  </div>

                  {/* Assigned Person */}
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-8 h-8 rounded-full flex items-center justify-center text-sm"
                      style={{ backgroundColor: `${assignedPerson.color}20` }}
                    >
                      {assignedPerson.avatar}
                    </div>
                    <div className="text-right">
                      <div className="text-xs font-medium text-gray-900">
                        {assignedPerson.name.split(' ')[0]}
                      </div>
                      <div className="flex items-center gap-1 mt-0.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${confidenceColor(assignment.confidence)}`} />
                        <span className="text-xs text-gray-500">
                          {(assignment.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Pending indicator */}
              {!assignment && (
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <div className="w-4 h-4 border-2 border-gray-300 border-t-transparent rounded-full animate-spin" />
                  <span>Analyzing...</span>
                </div>
              )}
            </div>
          );
        })}

        {filteredSessions.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Tv className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No sessions yet. Start the simulation to see live assignments.</p>
          </div>
        )}
      </div>
    </Card>
  );
};
