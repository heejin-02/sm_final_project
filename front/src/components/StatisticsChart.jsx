// src/components/StatisticsChart.jsx
import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function StatisticsChart({ data, period }) {
  if (!data) return null;

  // 차트용 데이터 변환
  const getChartData = () => {
    if (period === 'daily') {
      // 일간: 시간별 라인 차트
      return data.detailList?.map(item => ({
        time: item.datetime.split(' ')[1] || item.datetime, // 시간만 추출
        count: item.count || 0, // 실제 count 값 사용
        bugType: item.bugType,
        region: item.region
      })) || [];
    } else if (period === 'monthly') {
      // 월간: 주차별 바 차트
      return Object.entries(data.groupedData || {}).map(([key, group]) => ({
        period: group.title.replace('2025년 1월 ', '').replace(' (1일~7일)', '').replace(' (8일~14일)', '').replace(' (15일~21일)', '').replace(' (22일~31일)', ''),
        count: group.count,
        items: group.items.length
      }));
    } else if (period === 'yearly') {
      // 연간: 월별 바 차트
      return Object.entries(data.groupedData || {}).map(([key, group]) => ({
        month: group.title.replace('2024년 ', ''),
        count: group.count,
        items: group.items.length
      }));
    }
    return [];
  };

  // 해충 종류별 파이 차트 데이터
  const getPieData = () => {
    const bugCounts = {};
    
    if (period === 'daily' && data.detailList) {
      data.detailList.forEach(item => {
        bugCounts[item.bugType] = (bugCounts[item.bugType] || 0) + (item.count || 1);
      });
    } else if ((period === 'monthly' || period === 'yearly') && data.groupedData) {
      Object.values(data.groupedData).forEach(group => {
        group.items.forEach(item => {
          bugCounts[item.bugType] = (bugCounts[item.bugType] || 0) + 1;
        });
      });
    }

    return Object.entries(bugCounts).map(([name, value]) => ({ name, value }));
  };

  const chartData = getChartData();
  const pieData = getPieData();

  // 파이 차트 색상
  const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'];

  return (
    <div className="statistics-charts">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* 메인 차트 */}
        <div className="bordered-box">
          <h3 className="text-lg font-bold mb-4">
            📊 {period === 'daily' ? '시간별' : period === 'monthly' ? '주차별' : '월별'} 탐지 현황
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            {period === 'daily' ? (
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [value, '탐지 수']}
                  labelFormatter={(label) => `시간: ${label}`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#ef4444" 
                  strokeWidth={3}
                  dot={{ fill: '#ef4444', strokeWidth: 2, r: 6 }}
                  name="탐지 수"
                />
              </LineChart>
            ) : (
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={period === 'monthly' ? 'period' : 'month'} />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [value, '탐지 건수']}
                />
                <Legend />
                <Bar 
                  dataKey="count" 
                  fill="#3b82f6" 
                  name="탐지 건수"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>

        {/* 해충 종류별 파이 차트 */}
        <div className="bordered-box">
          <h3 className="text-lg font-bold mb-4">🐛 해충 종류별 분포</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name) => [value, '탐지 수']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* 구역별 히트맵 차트 (일간만) */}
        {period === 'daily' && (() => {
          // 구역별 데이터 집계
          const regionData = chartData.reduce((acc, item) => {
            const existing = acc.find(a => a.region === item.region);
            if (existing) {
              existing.count += item.count;
              existing.incidents += 1; // 탐지 횟수
            } else {
              acc.push({
                region: item.region,
                count: item.count,
                incidents: 1
              });
            }
            return acc;
          }, []);

          console.log('🏠 구역별 데이터:', regionData);

          return (
            <div className="bordered-box lg:col-span-2">
              <h3 className="text-lg font-bold mb-4">🏠 구역별 탐지 현황</h3>
              {regionData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart
                    data={regionData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="region"
                      tick={{ fontSize: 12, angle: -45, textAnchor: 'end' }}
                      height={60}
                    />
                    <YAxis
                      tickFormatter={(value) => `${value}건`}
                    />
                    <Tooltip
                      formatter={(value, name) => [
                        `${value}건`,
                        '탐지 수'
                      ]}
                      labelFormatter={(label) => `${label}`}
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #ccc',
                        borderRadius: '4px'
                      }}
                    />
                    <Bar
                      dataKey="count"
                      fill="#22c55e"
                      radius={[4, 4, 0, 0]}
                      stroke="#16a34a"
                      strokeWidth={1}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-48 text-gray-500">
                  구역별 데이터가 없습니다.
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
