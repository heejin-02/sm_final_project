// src/components/StatisticsChart.jsx
import React from 'react';
import DailyDetectionChart from './DailyDetectionChart';
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
      // 새로운 daily API 데이터 구조 사용
      if (data.hourlyStats) {
        return data.hourlyStats.map(item => ({
          time: `${item.hour}시`,
          count: item.count || 0
        }));
      }
      // 기존 구조 fallback
      return data.detailList?.map(item => ({
        time: item.datetime.split(' ')[1] || item.datetime, // 시간만 추출
        count: item.count || 0, // 실제 count 값 사용
        bugType: item.bugType,
        region: item.region
      })) || [];
    } else if (period === 'monthly') {
// 월간: API에서 주차별로 weeklyStats를 내려줄 때
    if (data.weeklyStats) {
      return data.weeklyStats.map(item => ({
        period: item.week,    // 예: '1주차'
        count:  item.count
      }));
    }
    // (기존 groupedData fallback, 필요 없으면 삭제)
    return Object.entries(data.groupedData || {}).map(([key, group]) => ({
      period: group.title.replace(/.*년 \d+월 /, ''),
      count:  group.count
    }));
    } else if (period === 'yearly') {
// 연간: API에서 monthlyCounts로 내려온 월별 데이터 사용
    if (data.monthlyStats) {
      return data.monthlyStats.map(item => ({
        month: `${item.month}월`,
        count: item.count
      }));
    }
    // (optional) 이전 groupedData 구조 fallback
    if (data.groupedData) {
      return Object.entries(data.groupedData).map(([_, group]) => ({
        month: group.title.replace(/.*년 /, ''),
        count: group.count
      }));
   }
    }
    return [];
  };

  // 해충 종류별 파이 차트 데이터
  const getPieData = () => {
    if (period === 'daily' && data.insectDistribution) {
      // 새로운 daily API 데이터 구조 사용
      return data.insectDistribution.map(item => ({
        name: item.insect,
        value: item.count
      }));
    }

     // 월간에도 같은 배열로 내려올 때
  if (period === 'monthly' && data.insectDistribution) {
    return data.insectDistribution.map(item => ({
      name: item.insect,
      value: item.count
    }));
  }

    if (period === 'yearly' && data.insectDistribution) {
    return data.insectDistribution.map(item => ({
      name: item.insect,
      value: item.count
    }));
  }

    // 기존 구조 fallback
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

  console.log('monthly chartData:', chartData);
  console.log('monthly pieData   :', pieData);


  // 파이 차트 색상
  const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'];

  return (
    <div className="statistics-charts">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* 메인 차트 */}
        <div className="bordered-box min-w-0 p-4 shadow-sm">
          {/* 1) 일별: DailyDetectionChart */}
          {period === 'daily' ? (
          <div className="">
            <h3 className="text-lg font-bold mb-4">⏰ 시간별 탐지 현황</h3>
            <div className="w-full h-64">
              <DailyDetectionChart hourlyStats={data.hourlyStats} />
            </div>
          </div>
          ) : (
            /* 2) 월간/연간: BarChart */
            <div>
              <h3 className="text-lg font-bold mb-4">
                📊 {period === 'monthly' ? '주차별' : '월별'} 탐지 현황
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={period === 'monthly' ? 'period' : 'month'} />
                  <YAxis />
                  <Tooltip formatter={(v) => [v, '탐지 건수']} />
                  <Legend />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* 해충 종류별 파이 차트 */}
        <div className="bordered-box min-w-0 p-4 shadow-sm">
          <h3 className="text-lg font-bold mb-4">🐛 해충 종류별 분포</h3>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
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
        </div>

        {/* 구역별 히트맵 차트 (일간만) */}
        {period === 'daily' && (() => {
          // 새로운 API 데이터 구조 사용
          let regionData = [];
          if (data.zoneStats && data.zoneStats.length > 0) {
            regionData = data.zoneStats.map(item => ({
              region: item.zone,
              count: item.count,
              incidents: item.count // 탐지 횟수와 동일하게 처리
            }));
          } else {
            // 기존 구조 fallback
            regionData = chartData.reduce((acc, item) => {
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
          }

          // console.log('구역별 데이터:', regionData);

          return (
            <div className="bordered-box lg:col-span-2 min-w-0">
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
