import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
} from 'recharts';

export default function DailyDetectionChart({ hourlyStats }) {
  // 1) 리스트 가드
  const list = Array.isArray(hourlyStats) ? hourlyStats : [];

  // 2) 차트 데이터 가공 (안전 변환 + 정렬)
  const chartData = useMemo(() => {
    if (list.length === 0) {
      // 0~23시 모두 0으로 채워 빈 상태를 만들면 축이 안정적으로 렌더됨
      return Array.from({ length: 24 }, (_, h) => ({ time: h, count: 0 }));
    }

    // 혹시 hour가 문자열("00","01")일 수 있으므로 숫자 변환
    const map = {};
    list.forEach((row) => {
      const h = Number(row?.hour);
      const c = Number(row?.count ?? 0);
      if (!Number.isNaN(h) && h >= 0 && h <= 23) {
        map[h] = (map[h] ?? 0) + c; // 같은 시각 여러 건이면 합산
      }
    });

    return Array.from({ length: 24 }, (_, h) => ({
      time: h,
      count: map[h] || 0,
    }));
  }, [list]);

  // 3) 렌더
  return (
    <div className='w-full h-full min-w-0'>
      {/* 부모가 실제 높이를 가지고 있어야 합니다. (예: 상위 div에 h-64 등) */}
      <ResponsiveContainer width='100%' height='100%'>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray='3 3' />
          <XAxis
            dataKey='time'
            type='number'
            domain={[0, 23]}
            ticks={Array.from({ length: 24 }, (_, i) => i)}
            tickFormatter={(t) => `${t}시`}
          />
          <YAxis allowDecimals={false} />
          <Tooltip
            formatter={(v) => [`${v}마리`, '탐지 해충 수']}
            labelFormatter={(l) => `시간: ${l}시`}
          />
          <Legend wrapperStyle={{ paddingTop: '10px', margin: 0 }} />
          <Line
            type='monotone'
            dataKey='count'
            name='탐지 해충 수'
            stroke='#ef4444'
            strokeWidth={3}
            dot={(props) => {
              const { cx, cy, payload } = props;
              if (payload.count > 0) {
                return (
                  <circle
                    key={`dot-${payload.time}`}
                    cx={cx}
                    cy={cy}
                    r={6}
                    strokeWidth={2}
                    fill='#ef4444'
                  />
                );
              }
              return null;
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
