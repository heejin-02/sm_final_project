// src/components/NotiFarmMap.jsx
import React from "react";

export default function NotiFarmMap({
  highlightRegion,  // 강조할 구역 ID (예: "A", "B", "C" 등)
  rows = 3,         // 세로 셀 개수
  cols = 3,         // 가로 셀 개수
  gap = 8,          // 셀 사이 간격(px)
}) {
  // 3x3 그리드의 구역 ID 생성 (A, B, C, D, E, F, G, H, I)
  const generateRegions = () => {
    const regions = [];
    const regionLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];
    
    for (let i = 0; i < rows * cols; i++) {
      regions.push({
        id: regionLabels[i] || `구역${i + 1}`,
        label: regionLabels[i] || `${i + 1}`,
        isHighlighted: regionLabels[i] === highlightRegion
      });
    }
    
    return regions;
  };

  const regions = generateRegions();

  return (
    <div className="noti-farm-map">
      <div
        className="farm-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gridTemplateRows: `repeat(${rows}, 1fr)`,
          gap: `${gap}px`,
          height: '100%',
          minHeight: '200px',
        }}
      >
        {regions.map((region) => (
          <div
            key={region.id}
            className={`farm-cell ${region.isHighlighted ? 'highlighted' : ''}`}
          >
            <span className="region-label">{region.label}구역</span>
            {region.isHighlighted && (
              <div className="alert-indicator">
                <span className="alert-icon">⚠️</span>
                <span className="alert-text">탐지됨</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
