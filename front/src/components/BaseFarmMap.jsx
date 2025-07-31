// src/components/BaseFarmMap.jsx
import React from "react";
import { scaleLinear } from "d3-scale";
import Loader from './Loader';

export default function BaseFarmMap({
  mode = "overview",           // "overview" | "highlight"
  data = [],                   // 히트맵용 데이터 [{ id, count }, ...]
  regions = [],                // 구역 데이터 [{ id, name }, ...]
  highlightRegion = null,      // 강조할 구역 이름
  loading = false,             // 로딩 상태
  rows = 3,                    // 세로 셀 개수
  cols = 3,                    // 가로 셀 개수
  gap = 8,                     // 셀 사이 간격(px)
  showHeatmap = false,         // 히트맵 표시 여부
  interactive = false,         // 클릭 가능 여부
  onCellClick = null,          // 클릭 핸들러
}) {
  
  // 로딩 중이거나 데이터가 없으면 로딩 표시
  if (loading) {
    return (
      <div className="base-farm-map">
        <div className="farm-map-loading">
          <Loader />
          <p>구역 정보를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  // 히트맵 색상 스케일 설정 (overview 모드용)
  let colorScale = null;
  if (mode === "overview" && showHeatmap && data.length > 0) {
    const counts = data.map(r => r.count || 0);
    const max = Math.max(...counts, 1);

    console.log(`🔥 BaseFarmMap: mode=${mode}, showHeatmap=${showHeatmap}, data.length=${data.length}`);
    console.log(`📊 Counts:`, counts, `Max: ${max}`);

    colorScale = scaleLinear()
      .domain([0, max / 2, max])
      .range(["#00AA00", "#FFFF00", "#FF0000"])
      .clamp(true);

    console.log(`🎨 ColorScale created:`, colorScale(0), colorScale(max/2), colorScale(max));
  }

  // 셀 데이터 준비
  const getCellData = () => {
    if (mode === "overview" && data.length > 0) {
      // 히트맵 모드: data 기준
      return data.map(item => ({
        id: item.id,
        name: item.name || `${item.id}번 구역`,
        count: item.count || 0,
        isHighlighted: false
      }));
    } else if (mode === "highlight" && regions.length > 0) {
      // 하이라이트 모드: regions 기준
      return regions.map(region => ({
        id: region.id,
        name: region.name || `${region.id}번 구역`,
        count: 0,
        isHighlighted: region.name === highlightRegion
      }));
    }
    return [];
  };

  const cellsData = getCellData();

  // 데이터가 없으면 빈 상태 표시
  if (cellsData.length === 0) {
    return (
      <div className="base-farm-map">
        <div className="farm-map-empty">
          <p>구역 데이터가 없습니다.</p>
        </div>
      </div>
    );
  }

  // 셀 스타일 계산
  const getCellStyle = (cell) => {
    const baseStyle = {
      cursor: interactive ? 'pointer' : 'default',
    };

    if (mode === "overview" && showHeatmap && colorScale) {
      // 히트맵 색상 적용 (opacity 추가로 가독성 향상)
      const originalColor = colorScale(cell.count);
      console.log(`🎨 Cell ${cell.id}: count=${cell.count}, color=${originalColor}`);

      // rgb를 rgba로 변환 (50% opacity)
      let backgroundColor = originalColor;
      if (typeof originalColor === 'string' && originalColor.startsWith('rgb(')) {
        // rgb(255, 102, 0) → rgba(255, 102, 0, 0.5)
        backgroundColor = originalColor.replace('rgb(', 'rgba(').replace(')', ', 0.5)');
      } else if (typeof originalColor === 'string' && originalColor.startsWith('#')) {
        // hex인 경우 alpha 추가
        backgroundColor = originalColor + '80';
      }

      baseStyle.backgroundColor = backgroundColor;
      baseStyle.color = cell.count > 0 ? '#000' : '#333';
      baseStyle.fontWeight = '600';
      baseStyle.textShadow = '0 1px 2px rgba(255, 255, 255, 0.8)';
    } else if (mode === "highlight" && cell.isHighlighted) {
      // 하이라이트 색상 적용 (opacity 추가)
      baseStyle.backgroundColor = '#ef444480'; // 50% opacity
      baseStyle.color = '#000';
      baseStyle.fontWeight = '700';
      baseStyle.border = '2px solid #dc2626';
      baseStyle.textShadow = '0 1px 2px rgba(255, 255, 255, 0.9)';
    } else {
      // 기본 색상
      baseStyle.backgroundColor = '#f3f4f6';
      baseStyle.color = '#374151';
      baseStyle.border = '1px solid #d1d5db';
    }

    return baseStyle;
  };

  // 셀 클릭 핸들러
  const handleCellClick = (cell) => {
    if (interactive && onCellClick) {
      onCellClick(cell.id);
    }
  };

  return (
    <div className="base-farm-map">
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
        {cellsData.map((cell) => (
          <div
            key={cell.id}
            className={`farm-cell ${cell.isHighlighted ? 'highlighted' : ''} ${interactive ? 'interactive' : ''}`}
            style={getCellStyle(cell)}
            onClick={() => handleCellClick(cell)}
          >
            <div className="cell-content">
              <span className="region-label">{cell.name}</span>
              
              {/* 히트맵 모드: 카운트 표시 */}
              {mode === "overview" && showHeatmap && (
                <span className="count-label">{cell.count}</span>
              )}
              
              {/* 하이라이트 모드: 알림 표시 */}
              {mode === "highlight" && cell.isHighlighted && (
                <div className="alert-indicator">
                  <span className="alert-icon">⚠️</span>
                  <span className="alert-text">탐지됨</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
