import React, { useMemo } from 'react';
import DailyDetectionChart from './DailyDetectionChart';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';

export default function MainBarChart({ data, period }) {
  const chartData = useMemo(() => {
    if (period === 'daily') {
      return data.hourlyStats?.map(item => ({
        time: `${item.hour}시`,
        count: item.count || 0
      })) || [];
    }

    if (period === 'monthly' && data.weeklyStats) {
      return data.weeklyStats.map(item => ({
        period: item.week,
        count: item.count
      }));
    }

    if (period === 'yearly' && data.monthlyStats) {
      return data.monthlyStats.map(item => ({
        month: `${item.month}월`,
        count: item.count
      }));
    }

    return [];
  }, [data, period]);

  if (period === 'daily') {
    return (
      <>
        <h3 className="text-lg font-bold mb-4">시간별 탐지 현황</h3>
        <div className="w-full h-64">
          <DailyDetectionChart hourlyStats={data.hourlyStats} />
        </div>
      </>
    );
  }

  return (
    <>
      <h3 className="text-lg font-bold mb-4">
        {period === 'monthly' ? '주차별 탐지 현황' : '월별 탐지 현황'}
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={period === 'monthly' ? 'period' : 'month'} />
          <YAxis />
          <Tooltip formatter={(value) => `${value}건`} />
          <Bar dataKey="count" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}
