import React from "react";
import { scaleLinear } from "d3-scale";
import { interpolateYlOrRd } from "d3-scale-chromatic";

export default function FarmMap({
  data,        // [{ id, count }, …]
  rows,        // 세로 셀 개수 (예: 3)
  cols,        // 가로 셀 개수 (예: 3)
  cellSize = 128,   // 각 셀 높이(px)
  gap = 4,          // 셀 사이격(px)
  onCellClick,      // 클릭 시 호출(id)
}) {
  if (!data || data.length === 0) return null;

  // 색상 스케일 설정
  const counts = data.map(r => r.count);
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  const colorScale = scaleLinear().domain([min, max]).range([0, 1]);

  return (
    <div
      className={`grid`}
      style={{
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        gap: `${gap}px`,
      }}
    >
      {data.map(region => {
        const ratio = colorScale(region.count);
        return (
          <div
            key={region.id}
            className="flex items-center justify-center text-white font-bold text-3xl rounded cursor-pointer"
            style={{
              backgroundColor: interpolateYlOrRd(ratio),
              height: `${cellSize}px`,
            }}
            onClick={() => onCellClick?.(region.id)}
          >
            {region.count}
          </div>
        );
      })}
    </div>
  );
}
