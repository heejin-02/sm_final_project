// src/components/RegionStackedBarChart.jsx
import React, { useEffect, useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { getTodayGreenhouseStats } from '../api/greenhouse';
import { useAuth } from '../contexts/AuthContext';
import {
  INSECT_COLOR,
  orderInsects,
  COLORS,
  normalizeInsect,
} from '../charts/constants';

// 레이아웃 상수
const BAR_SIZE = 28;
const ROW_GAP = 42; // px
const MARGIN = { top: 16, right: 0, left: 0, bottom: 20 };
const YPAD = { top: 12, bottom: 12 };
const LEGEND_H = 32;

export default function RegionStackedBarChart({ stats }) {
  const { user } = useAuth();
  const [zoneList, setZoneList] = useState([]);

  // 구역 목록 가져오기 (중복 제거)
  useEffect(() => {
    const fetchZones = async () => {
      try {
        const res = await getTodayGreenhouseStats(user?.selectedFarm?.farmIdx);
        const zones = (Array.isArray(res) ? res : [])
          .map((item) => item?.zone)
          .filter((z) => typeof z === 'string' && z.trim() !== '');
        setZoneList([...new Set(zones)]);
      } catch (err) {
        console.error('구역 리스트 불러오기 실패:', err);
        setZoneList([]);
      }
    };
    if (user?.selectedFarm?.farmIdx) fetchZones();
  }, [user?.selectedFarm?.farmIdx]);

  // 데이터 가공: 행(row) + 해충 타입(정해둔 우선순서로 정렬)
  const { regionData, insectTypes } = useMemo(() => {
    const zoneMap = {};
    const insectSet = new Set();

    // 구역 초기화
    zoneList.forEach((zone) => {
      const z = typeof zone === 'string' ? zone.trim() : '';
      if (z) zoneMap[z] = { region: z };
    });

    // 카운트 누적
    stats?.details?.forEach(({ greenhouse, insect }) => {
      const region = typeof greenhouse === 'string' ? greenhouse.trim() : '';
      const name = normalizeInsect(insect);
      if (!region || !name) return;
      if (!zoneMap[region]) zoneMap[region] = { region };
      zoneMap[region][name] = (zoneMap[region][name] ?? 0) + 1;
      insectSet.add(name);
    });

    const rows = Object.values(zoneMap)
      .filter((r) => !!r.region)
      .sort((a, b) => a.region.localeCompare(b.region, 'ko'));

    return {
      regionData: rows,
      insectTypes: orderInsects(Array.from(insectSet)), // ✅ 순서 고정
    };
  }, [stats, zoneList]);

  // 숫자 보정
  const safeRegionData = useMemo(() => {
    const rows = regionData.map((row) => {
      const r = { ...row, region: String(row.region || '') };
      insectTypes.forEach(
        (t) => (r[t] = Number.isFinite(Number(r[t])) ? Number(r[t]) : 0)
      );
      return r;
    });

    if (rows.length === 1) {
      rows.push({
        region: '__dummy__',
        ...Object.fromEntries(insectTypes.map((t) => [t, 0])),
      });
    }
    return rows;
  }, [regionData, insectTypes]);

  // 보여줄 데이터에서는 dummy 제외
  const chartData = safeRegionData.filter((r) => r.region !== '__dummy__');

  // 바 두께 기준 차트 높이 계산(범례 포함)
  const rowCount = safeRegionData.length;
  const chartHeight = useMemo(() => {
    if (!rowCount) return 180;
    const plot =
      MARGIN.top +
      MARGIN.bottom +
      YPAD.top +
      YPAD.bottom +
      rowCount * BAR_SIZE +
      Math.max(0, rowCount - 1) * ROW_GAP;
    return plot + LEGEND_H;
  }, [rowCount]);

  return (
    <>
      <h3 className='text-lg font-bold mb-4'>구역별 탐지 현황</h3>
      {rowCount > 0 ? (
        <ResponsiveContainer width='100%' height={chartHeight}>
          <BarChart
            data={chartData}
            layout='vertical'
            margin={MARGIN}
            barCategoryGap={ROW_GAP}
            barGap={0}
          >
            <CartesianGrid strokeDasharray='3 3' />
            <XAxis
              type='number'
              allowDecimals={false}
              tick={{ fontSize: 16 }}
              tickFormatter={(v) => `${v}건`}
            />
            <YAxis
              type='category'
              dataKey='region'
              width={90}
              interval={0}
              allowDuplicatedCategory={true}
              padding={YPAD}
              tickMargin={10}
            />
            <Tooltip formatter={(v) => `${v}건`} />
            <Legend
              verticalAlign='bottom'
              align='right'
              height={LEGEND_H}
              iconType='square'
              wrapperStyle={{
                paddingTop: '12px',
                fontSize: 16,
                marginLeft: 'auto',
                display: 'flex',
                justifyContent: 'flex-end',
                whiteSpace: 'nowrap',
              }}
            />

            {/* 같은 벌레 = 같은 색 */}
            {insectTypes.map((insect, idx) => (
              <Bar
                key={insect}
                dataKey={insect}
                stackId='a'
                fill={INSECT_COLOR[insect] ?? COLORS[idx % COLORS.length]}
                barSize={BAR_SIZE}
                // stroke="rgba(0,0,0,.12)" // 경계선이 필요하면 주석 해제
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className='flex items-center justify-center h-48 text-gray-500'>
          구역별 데이터가 없습니다.
        </div>
      )}
    </>
  );
}
