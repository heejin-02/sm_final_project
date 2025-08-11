// src/components/YearOverYearTable.jsx
import React from 'react';

const YearOverYearTable = ({ stats }) => {
  // 데이터 가공
  const insectNames = [...new Set(stats.predictedInsectTrends.map(p => p.insectName))];

  const insectDataMap = {};
  insectNames.forEach(name => {
    insectDataMap[name] = {
      spring: { count2024: 0, count2025: 0, predicted2026: 0 },
      summer: { count2024: 0, count2025: 0, predicted2026: 0 },
      fall:   { count2024: 0, count2025: 0, predicted2026: 0 },
      winter: { count2024: 0, count2025: 0, predicted2026: 0 },
    };
  });

  stats.predictedInsectTrends.forEach(p => {
    const season = p.season.toLowerCase();
    if (insectDataMap[p.insectName]) {
      insectDataMap[p.insectName][season] = {
        count2024: p.count2024,
        count2025: p.count2025,
        predicted2026: p.predicted2026,
      };
    }
  });

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
                <th>2024</th><th>2025</th><th style={{ backgroundColor: '#f9f9f9' }}>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th style={{ backgroundColor: '#f9f9f9' }}>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th style={{ backgroundColor: '#f9f9f9' }}>2026 <br/>예측</th>
                <th>2024</th><th>2025</th><th style={{ backgroundColor: '#f9f9f9' }}>2026 <br/>예측</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(insectDataMap).map(([name, seasons]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{seasons.spring.count2024}</td>
                  <td>{seasons.spring.count2025}</td>
                  <td style={{ backgroundColor: '#f9f9f9' }}>{seasons.spring.predicted2026}</td>

                  <td>{seasons.summer.count2024}</td>
                  <td>{seasons.summer.count2025}</td>
                  <td style={{ backgroundColor: '#f9f9f9' }}>{seasons.summer.predicted2026}</td>

                  <td>{seasons.fall.count2024}</td>
                  <td>{seasons.fall.count2025}</td>
                  <td style={{ backgroundColor: '#f9f9f9' }}>{seasons.fall.predicted2026}</td>

                  <td>{seasons.winter.count2024}</td>
                  <td>{seasons.winter.count2025}</td>
                  <td style={{ backgroundColor: '#f9f9f9' }}>{seasons.winter.predicted2026}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default YearOverYearTable;
