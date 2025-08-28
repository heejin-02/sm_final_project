// src/components/NotiFarmMap.jsx
import Loader from './Loader';
import { useRegions } from '../hooks/useRegions';

export default function NotiFarmMap({
  highlightRegion,               // 강조할 구역 이름 (fallback)
  highlightGhIdx = null,         // 강조할 구역 ID (우선순위 높음)
  regions: propRegions = [],     // 외부에서 전달받은 구역 데이터 (선택사항)
  loading: propLoading = false,  // 외부 로딩 상태 (선택사항)
  rows = null,                   // 세로 셀 개수 (null이면 자동 계산)
  cols = null,                   // 가로 셀 개수 (null이면 자동 계산)
  gap = 0,                       // 셀 사이 간격(px)
  useApiData = true,             // API 데이터 사용 여부 (기본: true)
}) {
  // API 데이터 사용 시 useRegions Hook 사용
  const { regions: apiRegions, loading: apiLoading, error } = useRegions();

  // 사용할 데이터와 로딩 상태 결정
  const regions = useApiData ? apiRegions : propRegions;
  const loading = useApiData ? apiLoading : propLoading;

  // 에러 처리
  if (useApiData && error) {
    return (
      <div className="noti-farm-map">
        <div className="farm-map-loading">
          <p className="text-red-500">구역 정보를 불러오는데 실패했습니다.</p>
          <p className="text-sm text-gray-500">잠시 후 다시 시도해주세요.</p>
        </div>
      </div>
    );
  }

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

  // 동적 그리드 크기 계산
  const calculateGridSize = (itemCount) => {
    if (itemCount <= 4) return { rows: 2, cols: 2 };
    if (itemCount <= 6) return { rows: 2, cols: 3 };
    if (itemCount <= 9) return { rows: 3, cols: 3 };
    if (itemCount <= 12) return { rows: 3, cols: 4 };
    return { rows: 4, cols: 4 }; // 최대 16개
  };

  const dynamicGrid = calculateGridSize(regions.length);
  const finalRows = rows || dynamicGrid.rows;
  const finalCols = cols || dynamicGrid.cols;

  // 실제 구역 데이터 처리 (개선된 하이라이트 로직)
  const regionsData = regions.map(region => {
    let isHighlighted = false;

    // ghIdx 우선 비교 (더 정확함)
    if (highlightGhIdx) {
      isHighlighted = region.id === highlightGhIdx;
    }
    // ghName으로 비교 (fallback)
    else if (highlightRegion) {
      isHighlighted = region.name === highlightRegion;
    }

    return {
      id: region.id,
      name: region.name,
      isHighlighted
    };
  });

  return (
    <div className="noti-farm-map">
      <div
        className="farm-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${finalCols}, 1fr)`,
          gridTemplateRows: `repeat(${finalRows}, 1fr)`,
          gap: `${gap}px`,
          height: '100%',
          minHeight: '240px',
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
                {/* <span className="alert-icon">⚠️</span> */}
                <span className="alert-text">탐지됨</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
