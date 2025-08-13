// src/components/YearOverYearTable.jsx
import React, { useMemo } from 'react';

// 시즌 키 정규화 맵
const SEASON_KEYS = {
  spring: 'spring', summer: 'summer', fall: 'fall', autumn: 'fall', winter: 'winter',
  '봄': 'spring', '여름': 'summer', '가을': 'fall', '겨울': 'winter',
};

const normalizeSeason = (s) => {
  if (!s) return null;
  const k = String(s).trim().toLowerCase();
  return SEASON_KEYS[k] ?? null;
};

export default function YearOverYearTable({ stats }) {
  // 리스트 안전 가드
  const list = Array.isArray(stats?.predictedInsectTrends)
    ? stats.predictedInsectTrends
    : [];

  // 데이터 없으면 빈 상태 표시
  if (list.length === 0) {
    return (
      <div className="mt-8">
        <div className="bordered-box p-4 text-gray-500">예측 데이터가 없습니다.</div>
      </div>
    );
  }

  // 가공은 메모이즈
  const { insectDataMap, insectNames } = useMemo(() => {
    const names = [...new Set(list.map(p => p?.insectName).filter(Boolean))];

    const base = () =>
      ({ count2024: 0, count2025: 0, predicted2026: 0 });

    const map = {};
    names.forEach(name => {
      map[name] = {
        spring: base(),
        summer: base(),
        fall:   base(),
        winter: base(),
      };
    });

    list.forEach(p => {
      if (!p) return;
      const seasonKey = normalizeSeason(p.season);
      if (!seasonKey) return;
      const name = p.insectName;
      if (!map[name]) return;

      map[name][seasonKey] = {
        count2024: Number(p.count2024 ?? 0),
        count2025: Number(p.count2025 ?? 0),
        predicted2026: Number(p.predicted2026 ?? 0),
      };
    });

    return { insectDataMap: map, insectNames: names };
  }, [list]);

  return (
    <div className="mt-8">
      <div className="bordered-box">
        <h3 className="text-lg font-bold mb-4">내년 해충 발생 예측</h3>
        <div className="table-overflow scrl-custom">
          <table className="table border">
            <thead>
              <tr>
                <th rowSpan="2">해충 종류</th>
                <th colSpan="3">봄</th>
                <th colSpan="3">여름</th>
                <th colSpan="3">가을</th>
                <th colSpan="3">겨울</th>
              </tr>
              <tr>
                <th>2024</th><th>2025</th><th>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th>2026 <br/>예측</th>
              </tr>
            </thead>
            <tbody>
              {insectNames.map(name => {
                const s = insectDataMap[name];
                return (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{s.spring.count2024}</td>
                    <td>{s.spring.count2025}</td>
                    <td style={{ backgroundColor: 'aliceblue' }}>{s.spring.predicted2026}</td>

                    <td>{s.summer.count2024}</td>
                    <td>{s.summer.count2025}</td>
                    <td style={{ backgroundColor: 'aliceblue' }}>{s.summer.predicted2026}</td>

                    <td>{s.fall.count2024}</td>
                    <td>{s.fall.count2025}</td>
                    <td style={{ backgroundColor: 'aliceblue' }}>{s.fall.predicted2026}</td>

                    <td>{s.winter.count2024}</td>
                    <td>{s.winter.count2025}</td>
                    <td style={{ backgroundColor: 'aliceblue' }}>{s.winter.predicted2026}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
