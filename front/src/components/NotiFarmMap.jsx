// src/components/NotiFarmMap.jsx
import Loader from './Loader';

export default function NotiFarmMap({
  highlightRegion,  // 강조할 구역 이름 (예: "문앞", "00밭", "1번 레인" 등)
  regions = [],     // 구역 데이터 배열 [{ id, name }, ...]
  loading = false,  // 로딩 상태
  rows = 3,         // 세로 셀 개수
  cols = 3,         // 가로 셀 개수
  gap = 8,          // 셀 사이 간격(px)
}) {
  // 로딩 중이거나 구역 데이터가 없으면 로딩 표시
  if (loading || !regions || regions.length === 0) {
    return (
      <div className="noti-farm-map">
        <div className="farm-map-loading">
          <Loader />
          <p>구역 정보를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  // 실제 구역 데이터만 사용
  const regionsData = regions.map(region => ({
    id: region.id,
    name: region.name,
    isHighlighted: region.name === highlightRegion
  }));

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
        {regionsData.map((region) => (
          <div
            key={region.id}
            className={`farm-cell ${region.isHighlighted ? 'highlighted' : ''}`}
          >
            <span className="region-label">{region.name}</span>
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
