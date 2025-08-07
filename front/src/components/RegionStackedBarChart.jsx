import React, { useEffect, useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';
import { getTodayGreenhouseStats } from '../api/greenhouse';
import { useAuth } from '../contexts/AuthContext';

const COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e',
  '#3b82f6', '#8b5cf6', '#14b8a6', '#f43f5e'
];

export default function RegionStackedBarChart({ data, period }) {
  const { user } = useAuth();
  const [zoneList, setZoneList] = useState([]);

  useEffect(() => {
    const fetchZones = async () => {
      try {
        const res = await getTodayGreenhouseStats(user?.selectedFarm?.farmIdx);
        setZoneList(res.map(item => item.zone));
      } catch (err) {
        console.error('구역 리스트 불러오기 실패:', err);
        setZoneList([]);
      }
    };

    if (user?.selectedFarm?.farmIdx) {
      fetchZones();
    }
  }, [user]);

  const { regionData, insectTypes } = useMemo(() => {
    const zoneMap = {};
    const insectSet = new Set();

    zoneList.forEach(zone => {
      zoneMap[zone] = { region: zone };
    });

    data?.details?.forEach(({ greenhouse, insect }) => {
      if (!zoneMap[greenhouse]) {
        zoneMap[greenhouse] = { region: greenhouse };
      }
      zoneMap[greenhouse][insect] = (zoneMap[greenhouse][insect] || 0) + 1;
      insectSet.add(insect);
    });

    return {
      regionData: Object.values(zoneMap),
      insectTypes: Array.from(insectSet)
    };
  }, [data, zoneList]);

  return (
    <>
      <h3 className="text-lg font-bold mb-4">구역별 탐지 현황</h3>
      {regionData.length > 0 ? (
        <ResponsiveContainer width="100%" height={regionData.length * 50}>
          <BarChart
            data={regionData}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              tick={{ fontSize: 16 }}
              tickFormatter={(v) => `${v}건`}
            />
            <YAxis
              type="category"
              dataKey="region"
              width={100}
              tick={{ fontSize: 16 }}
            />
            <Tooltip formatter={(value) => `${value}건`} />
            <Legend
              verticalAlign="bottom"
              align="right"
              iconType="square"
              wrapperStyle={{
                paddingTop: '12px',
                fontSize: 16,
                marginLeft: 'auto',
                display: 'flex',
                justifyContent: 'flex-end',
              }}
            />
            {insectTypes.map((insect, idx) => (
              <Bar
                key={insect}
                dataKey={insect}
                stackId="a"
                fill={COLORS[idx % COLORS.length]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-500">
          구역별 데이터가 없습니다.
        </div>
      )}
    </>
  );
}
