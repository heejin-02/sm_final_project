import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart, CartesianGrid,
  XAxis, YAxis, Tooltip,
  Legend, Line
} from 'recharts';

export default function DailyDetectionChart({ hourlyStats }) {
  const chartData = useMemo(() => {
    const map = {};
    hourlyStats.forEach(({ hour, count }) => {
      map[+hour] = count;
    });
    return Array.from({ length: 24 }, (_, h) => ({
      time: h,
      count: map[h] || 0
    }));
  }, [hourlyStats]);

  return (
    // 이 div가 “h-64” 를 물려받아 높이가 256px 이 되도록 합니다.
    <div className="w-full h-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="time"
            type="number"
            domain={[0, 23]}
            ticks={Array.from({ length: 24 }, (_, i) => i)}
            tickFormatter={t => `${t}시`}
          />
          <YAxis />
          <Tooltip
            formatter={v => [`${v}마리`, '탐지 해충 수']}
            labelFormatter={l => `시간: ${l}시`}
          />
          <Legend
            wrapperStyle={{
              paddingTop: '10px',
              margin: '0',
            }}
          />
          <Line
            type="monotone"
            dataKey="count"
            name="탐지 해충 수"
            stroke="#ef4444"
            strokeWidth={3}
            dot={props => {
              const { cx, cy, payload } = props;
              if (payload.count > 0) {
                // 각 dot마다 고유 key를 달아 줍니다
                return <circle key={`dot-${payload.time}`} cx={cx} cy={cy} r={6} strokeWidth={2} fill="#ef4444" />;
              }
              return null;
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
