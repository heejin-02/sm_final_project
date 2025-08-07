// src/components/YearOverYearTable.jsx
import React from 'react';

export default function YearOverYearTable({ predictions }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="mt-8">
        <div className="bordered-box">
          <p>📉 예측 데이터가 없습니다.</p>
        </div>
      </div>
    );
  }

  const sample = predictions[0];
  const [baseYear, baseMonth] = sample.month.split('-');

  const colHeaders = [
    `${baseYear - 1}.${baseMonth}`,
    `${baseYear}.${baseMonth}`,
    `${Number(baseYear) + 1} 예측`
  ];

  return (
    <div className="mt-8">
      <div className="bordered-box">
        <h3 className="text-lg font-bold mb-4">내년 해충 발생 예측</h3>
        <div className="overflow-x-auto">
          <table className="table border text-center">
            <thead>
              <tr>
                <th className="">해충 종류</th>
                {colHeaders.map((label, idx) => (
                  <th key={idx} className="">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {predictions.map((row, idx) => (
                <tr key={idx}>
                  <td className="">{row.insectName}</td>
                  <td className="">{row.count2024}</td>
                  <td className="">{row.count2025}</td>
                  <td className="">{row.predicted2026}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
