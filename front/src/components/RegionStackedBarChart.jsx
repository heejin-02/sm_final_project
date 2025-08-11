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


// 컴포넌트 밖: 순수 상수만
const BAR_SIZE = 30;
const ROW_GAP  = 6;
const MARGIN   = { top: 16, right: 0, left: 0, bottom: 20 };
const YPAD     = { top: 12, bottom: 12 };
const LEGEND_H = 32;

const COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e',
  '#3b82f6', '#8b5cf6', '#14b8a6', '#f43f5e'
];

export default function RegionStackedBarChart({ stats, period }) {
  const { user } = useAuth();
  const [zoneList, setZoneList] = useState([]);

  useEffect(() => {
    const fetchZones = async () => {
      try {
        const res = await getTodayGreenhouseStats(user?.selectedFarm?.farmIdx);
        const zones = (Array.isArray(res) ? res : [])
          .map(item => item?.zone)
          .filter(z => typeof z === 'string' && z.trim() !== '');
        setZoneList([...new Set(zones)]);
      } catch (err) {
        console.error('구역 리스트 불러오기 실패:', err);
        setZoneList([]);
      }
    };
    if (user?.selectedFarm?.farmIdx) fetchZones();
  }, [user?.selectedFarm?.farmIdx]);

  // 1) 원본 -> regionData, insectTypes
  const { regionData, insectTypes } = useMemo(() => {
    const zoneMap = {};
    const insectSet = new Set();

    zoneList.forEach(zone => {
      const z = typeof zone === 'string' ? zone.trim() : '';
      if (z) zoneMap[z] = { region: z };
    });

    stats?.details?.forEach(({ greenhouse, insect }) => {
      const region = typeof greenhouse === 'string' ? greenhouse.trim() : '';
      const name   = typeof insect === 'string' ? insect.trim() : '';
      if (!region || !name) return;
      if (!zoneMap[region]) zoneMap[region] = { region };
      zoneMap[region][name] = (zoneMap[region][name] ?? 0) + 1;
      insectSet.add(name);
    });

    const rows = Object.values(zoneMap)
      .filter(r => !!r.region)
      .sort((a, b) => a.region.localeCompare(b.region, 'ko'));

    return { regionData: rows, insectTypes: Array.from(insectSet) };
  }, [stats, zoneList]);

  // 2) 안전 숫자 변환
  const safeRegionData = useMemo(() => {
    return regionData.map(row => {
      const r = { ...row };
      insectTypes.forEach(type => { r[type] = Number(r[type] ?? 0); });
      return r;
    });
  }, [regionData, insectTypes]);

  // 높이 계산
  const rowCount = safeRegionData.length;
  const chartHeight = useMemo(() => {
    if (!rowCount) return 180;
    const plot = MARGIN.top + MARGIN.bottom + YPAD.top + YPAD.bottom
      + rowCount * BAR_SIZE + Math.max(0, rowCount - 1) * ROW_GAP;
    return plot + LEGEND_H; // ← 범례 자리까지 포함
  }, [rowCount]);

  return (
    <>
      <h3 className="text-lg font-bold mb-4">구역별 탐지 현황</h3>
      {rowCount > 0 ? (
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart
            data={safeRegionData}
            layout="vertical"
            margin={MARGIN}
            barCategoryGap={ROW_GAP}
            barGap={0}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              allowDecimals={false}
              tick={{ fontSize: 16 }}
              tickFormatter={(v) => `${v}건`}
            />
            <YAxis
              type="category"
              dataKey="region"
              width={100}
              tick={{ fontSize: rowCount > 8 ? 12 : 16 }}
              interval={0}
              allowDuplicatedCategory={false}
              padding={YPAD}
            />
            <Tooltip formatter={(v) => `${v}건`} />
            <Legend
              verticalAlign="bottom"
              align="right"
              height={LEGEND_H}       
              iconType="square"
              wrapperStyle={{
                paddingTop: '12px',
                fontSize: 16,
                marginLeft: 'auto',
                display: 'flex',
                justifyContent: 'flex-end',
                whiteSpace: 'nowrap' 
              }}
            />
            {insectTypes.map((insect, idx) => (
              <Bar
                key={insect}
                dataKey={insect}
                stackId="a"
                fill={COLORS[idx % COLORS.length]}
                barSize={BAR_SIZE}  // 바 두께 고정
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
