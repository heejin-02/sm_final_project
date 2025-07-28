import React from "react";
import { scaleLinear } from "d3-scale";
import { rgb } from "d3-color";

export default function FarmMap({
  data,        // [{ id, count }, …]
  rows,        // 세로 셀 개수 (예: 3)
  cols,        // 가로 셀 개수 (예: 3)
  gap = 4,          // 셀 사이격(px)
  onCellClick,      // 클릭 시 호출(id)
}) {
  if (!data || data.length === 0) return null;

  // 색상 스케일 설정
  // 1) 최대값 계산 (0은 고정)
  const counts = data.map(r => r.count);
  const max = Math.max(...counts, 1); // all zero일 때 분모 0 방지

  // 3단계 그라디언트 스케일: 0→max/2→max
  const colorScale = scaleLinear()
    .domain([0, max / 2, max])
    .range(["#00AA00" /* 진한 연두 */, "#FFFF00" /* 노랑 */, "#FF0000" /* 빨강 */])
    .clamp(true);

  return (
    <div
      className={`grid`}
      style={{
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gridTemplateRows: `repeat(${rows}, 1fr)`,
        gap: `${gap}px`,
        height: "100%",
      }}
    >
      {data.map((r, i) => {
        const c = rgb(colorScale(r.count));          // d3-color 로 파싱
        c.opacity = 0.5;              // α = 0.6 (60%)
        const bg = c.formatRgb();     // "rgba(r,g,b,0.6)"
        return (
          <div
            key={r.id ?? i}
            className="flex items-center justify-center text-white font-bold text-3xl rounded cursor-pointer"
            style={{
              backgroundColor: bg,
            }}
            onClick={() => onCellClick?.(r.id)}
          >
            {r.count}
          </div>
        );
      })}
    </div>
  );
}
