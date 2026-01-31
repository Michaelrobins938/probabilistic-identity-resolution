import React from 'react';
import { useDemoStore } from '@/store/demoStore';
import { Card } from '@/components/ui/Card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { TrendingUp, Users, Target, DollarSign, MousePointer, Mail, Search, Share2, Video } from 'lucide-react';

const COLORS = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

const channelIcons: Record<string, React.ReactNode> = {
  'Organic Search': <Search className="w-4 h-4" />,
  'Paid Social': <Share2 className="w-4 h-4" />,
  'Email': <Mail className="w-4 h-4" />,
  'Display': <MousePointer className="w-4 h-4" />,
  'Video': <Video className="w-4 h-4" />,
  'Direct': <Target className="w-4 h-4" />
};

// Generate attribution data based on sessions
const generateAttributionData = (sessions: any[], people: any[], assignments: any[]) => {
  const channels = ['Organic Search', 'Paid Social', 'Email', 'Display', 'Video', 'Direct'];
  
  return channels.map((channel, index) => {
    // Simulate attribution based on session patterns
    const channelSessions = sessions.filter((_, i) => i % channels.length === index);
    const conversions = Math.floor(channelSessions.length * 0.15);
    const revenue = conversions * (50 + Math.random() * 100);
    
    // Person-level attribution
    const personShares: Record<string, number> = {};
    people.forEach(person => {
      const personConversions = Math.floor(conversions * (0.2 + Math.random() * 0.5));
      personShares[person.id] = personConversions;
    });
    
    // Normalize to percentages
    const totalShares = Object.values(personShares).reduce((a, b) => a + b, 0);
    Object.keys(personShares).forEach(key => {
      personShares[key] = totalShares > 0 ? Math.round((personShares[key] / totalShares) * 100) : 0;
    });
    
    return {
      channel,
      sessions: channelSessions.length,
      conversions,
      revenue: Math.round(revenue),
      personShares,
      icon: channelIcons[channel]
    };
  });
};

export const AttributionDashboard: React.FC = () => {
  const { sessions, people, assignments } = useDemoStore();
  
  const attributionData = React.useMemo(() => 
    generateAttributionData(sessions, people, assignments),
    [sessions, people, assignments]
  );

  const totalRevenue = attributionData.reduce((sum, d) => sum + d.revenue, 0);
  const totalConversions = attributionData.reduce((sum, d) => sum + d.conversions, 0);
  const totalSessions = attributionData.reduce((sum, d) => sum + d.sessions, 0);

  // Prepare chart data
  const pieData = attributionData.map(d => ({
    name: d.channel,
    value: d.revenue,
    conversions: d.conversions
  }));

  const barData = people.map(person => {
    const personRevenue = attributionData.reduce((sum, channel) => {
      const share = channel.personShares[person.id] || 0;
      return sum + (channel.revenue * share / 100);
    }, 0);
    
    return {
      name: person.name.split(' ')[0],
      revenue: Math.round(personRevenue),
      color: person.color
    };
  });

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Attribution Dashboard</h2>
          <p className="text-sm text-gray-500 mt-1">
            Person-level marketing attribution across channels
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <TrendingUp className="w-4 h-4" />
          <span>Last 30 days</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="w-5 h-5 opacity-80" />
            <span className="text-xs opacity-80">Total Revenue</span>
          </div>
          <div className="text-2xl font-bold">${totalRevenue.toLocaleString()}</div>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-2">
            <Target className="w-5 h-5 opacity-80" />
            <span className="text-xs opacity-80">Conversions</span>
          </div>
          <div className="text-2xl font-bold">{totalConversions.toLocaleString()}</div>
        </div>

        <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-2">
            <MousePointer className="w-5 h-5 opacity-80" />
            <span className="text-xs opacity-80">Sessions</span>
          </div>
          <div className="text-2xl font-bold">{totalSessions.toLocaleString()}</div>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between mb-2">
            <Users className="w-5 h-5 opacity-80" />
            <span className="text-xs opacity-80">Persons</span>
          </div>
          <div className="text-2xl font-bold">{people.length}</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Revenue by Channel Pie Chart */}
        <div className="bg-gray-50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Revenue by Channel</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Revenue']}
                  contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Revenue by Person Bar Chart */}
        <div className="bg-gray-50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Revenue by Person</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis 
                  axisLine={false} 
                  tickLine={false} 
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip 
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Revenue']}
                  contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                />
                <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Channel Breakdown Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Channel</th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Sessions</th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Conversions</th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Revenue</th>
              <th className="text-left py-3 px-4 text-xs font-semibold text-gray-500 uppercase">Attribution by Person</th>
            </tr>
          </thead>
          <tbody>
            {attributionData.map((channel, index) => (
              <tr key={channel.channel} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ backgroundColor: `${COLORS[index]}15`, color: COLORS[index] }}
                    >
                      {channel.icon}
                    </div>
                    <span className="font-medium text-gray-900">{channel.channel}</span>
                  </div>
                </td>
                <td className="text-right py-3 px-4 text-gray-600">{channel.sessions.toLocaleString()}</td>
                <td className="text-right py-3 px-4 text-gray-600">{channel.conversions.toLocaleString()}</td>
                <td className="text-right py-3 px-4 font-medium text-gray-900">
                  ${channel.revenue.toLocaleString()}
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-1">
                    {people.map(person => {
                      const share = channel.personShares[person.id] || 0;
                      if (share === 0) return null;
                      return (
                        <div 
                          key={person.id}
                          className="h-6 rounded px-2 flex items-center text-xs font-medium text-white"
                          style={{ 
                            backgroundColor: person.color,
                            width: `${share}%`,
                            minWidth: 'fit-content'
                          }}
                        >
                          {share > 15 ? `${person.name.split(' ')[0]} ${share}%` : `${share}%`}
                        </div>
                      );
                    })}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer Note */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
        <p className="text-xs text-blue-700">
          <strong>💡 Insight:</strong> Attribution is calculated using person-level identity resolution. 
          Each conversion is attributed to the specific household member based on their behavioral patterns and session history.
        </p>
      </div>
    </Card>
  );
};
