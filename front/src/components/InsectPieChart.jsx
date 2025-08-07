import React, { useMemo } from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'];

export default function InsectPieChart({ data }) {
  const pieData = useMemo(() => (
    data.insectDistribution?.map(item => ({
      name: item.insect,
      value: item.count
    })) || []
  ), [data]);

  return (
    <>
      <h3 className="text-lg font-bold mb-4">해충 종류별 분포</h3>
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              outerRadius={80}
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              dataKey="value"
            >
              {pieData.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${value}건`} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
