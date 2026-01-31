import React from 'react';
import { useDemoStore } from '@/store/demoStore';
import { Card } from '@/components/ui/Card';
import { TV, Smartphone, Tablet, Monitor } from 'lucide-react';

const deviceIcons: Record<string, React.ReactNode> = {
  'TV': <TV className="w-4 h-4" />,
  'Mobile': <Smartphone className="w-4 h-4" />,
  'Tablet': <Tablet className="w-4 h-4" />,
  'Desktop': <Monitor className="w-4 h-4" />
};

export const HouseholdSimulator: React.FC = () => {
  const { people, selectedPerson, selectPerson, assignments, isRunning } = useDemoStore();

  // Calculate stats for each person
  const getPersonStats = (personId: string) => {
    const personAssignments = assignments.filter(a => a.personId === personId);
    const avgConfidence = personAssignments.length > 0
      ? personAssignments.reduce((sum, a) => sum + a.confidence, 0) / personAssignments.length
      : 0;
    
    return {
      sessionCount: personAssignments.length,
      avgConfidence: avgConfidence * 100
    };
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">Household Members</h2>
        {isRunning && (
          <span className="flex items-center text-sm text-green-600">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
            Live Simulation
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {people.map((person) => {
          const stats = getPersonStats(person.id);
          const isSelected = selectedPerson === person.id;
          
          return (
            <div
              key={person.id}
              onClick={() => selectPerson(isSelected ? null : person.id)}
              className={`
                relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-300
                ${isSelected 
                  ? 'border-opacity-100 shadow-lg scale-105' 
                  : 'border-opacity-40 hover:border-opacity-70'
                }
              `}
              style={{ 
                borderColor: person.color,
                backgroundColor: isSelected ? `${person.color}10` : 'white'
              }}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div 
                    className="w-12 h-12 rounded-full flex items-center justify-center text-2xl"
                    style={{ backgroundColor: `${person.color}20` }}
                  >
                    {person.avatar}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{person.name}</h3>
                    <p className="text-xs text-gray-500 capitalize">{person.ageGroup}</p>
                  </div>
                </div>
                
                {stats.sessionCount > 0 && (
                  <div 
                    className="px-2 py-1 rounded-full text-xs font-medium"
                    style={{ 
                      backgroundColor: person.color,
                      color: 'white'
                    }}
                  >
                    {stats.sessionCount} sessions
                  </div>
                )}
              </div>

              {/* Details */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2 text-gray-600">
                  <span className="font-medium">Devices:</span>
                  <div className="flex items-center gap-1">
                    {person.devices.map(d => (
                      <span key={d} className="text-gray-400" title={d}>
                        {deviceIcons[d]}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1">
                  {person.genres.slice(0, 3).map(genre => (
                    <span 
                      key={genre}
                      className="px-2 py-0.5 bg-gray-100 rounded text-xs text-gray-600"
                    >
                      {genre}
                    </span>
                  ))}
                </div>
                
                <p className="text-xs text-gray-500">
                  <span className="font-medium">Active:</span> {person.timeOfDay}
                </p>
              </div>

              {/* Confidence indicator */}
              {stats.avgConfidence > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">Avg Confidence</span>
                    <span className="font-medium" style={{ color: person.color }}>
                      {stats.avgConfidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ 
                        width: `${stats.avgConfidence}%`,
                        backgroundColor: person.color
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <p className="mt-4 text-xs text-gray-500 text-center">
        Click on a household member to filter sessions
      </p>
    </Card>
  );
};
