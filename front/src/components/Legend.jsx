// 지도 시각화 색상 범례(gradient bar + 텍스트)
import React from "react";

export default function Legend({ min, max, steps = 5 }) {
  // steps개로 나눠 그라디언트 막대+텍스트
  const range = max - min;
  const stepSize = range / (steps - 1);
  const colors = Array.from({length:steps}, (_,i) => {
    // d3 interpolate 대신 Tailwind 색상 써도 무방
    const ratio = i/(steps-1);
    return { label: Math.round(min + stepSize*i), color: `rgba(255,0,0,${ratio})` };
  });

  return (
    <div className="absolute top-1 left-1 bg-white rounded">
      {colors.map(c => (
        <div key={c.label} className="flex flex-col items-center">
          <div 
            className="w-6 h-2 rounded" 
            style={{ backgroundColor: c.color }} 
          />
          <span>{c.label}</span>
        </div>
      ))}
      <span className="ml-2">해충 수</span>
    </div>
  );
}
