// components/InsectPieChart.jsx
import React, { useMemo } from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import {
  INSECT_COLOR,
  orderInsects,
  COLORS,
  normalizeInsect,
} from '../charts/constants';

export default function InsectPieChart({ stats }) {
  const raw = Array.isArray(stats?.insectDistribution)
    ? stats.insectDistribution
    : [];

  // 1) 이름 정규화
  const normalized = useMemo(
    () =>
      raw
        .map(({ insect, count }) => ({
          name: normalizeInsect(insect),
          value: Number(count ?? 0),
        }))
        .filter((d) => d.name),
    [raw]
  );

  // 2) 고정 우선순서로 정렬
  const insects = useMemo(
    () => orderInsects(normalized.map((d) => d.name)),
    [normalized]
  );

  // 3) 정렬 순서대로 pie 데이터 구성
  const byName = useMemo(
    () => new Map(normalized.map((d) => [d.name, d.value])),
    [normalized]
  );
  const pieData = useMemo(
    () => insects.map((name) => ({ name, value: byName.get(name) ?? 0 })),
    [insects, byName]
  );

  // 디버깅(원인 추적용): 둘 다 같은 순서인지 확인
  // console.log('PIE order:', pieData.map(d => d.name));

  return (
    <>
      <h3 className='text-lg font-bold mb-4'>해충 종류별 분포</h3>
      <div className='w-full h-64'>
        <ResponsiveContainer width='100%' height='100%'>
          <PieChart>
            <Pie
              data={pieData}
              cx='50%'
              cy='50%'
              outerRadius={80}
              dataKey='value'
              labelLine={false}
              label={({ name, value, percent }) =>
                value > 0 ? `${name} ${(percent * 100).toFixed(0)}%` : ''
              }
            >
              {pieData.map((d, i) => (
                <Cell
                  key={d.name}
                  fill={INSECT_COLOR[d.name] ?? COLORS[i % COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip formatter={(v) => `${v}마리`} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
