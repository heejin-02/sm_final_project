// src/hooks/useRegions.js
import { useState, useEffect } from 'react';

export function useRegions() {
  const [regions, setRegions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // TODO: 실제 API 호출로 교체
    const fetchRegions = async () => {
      try {
        setLoading(true);
        
        // 임시 더미 데이터 - 실제로는 API에서 가져옴
        const dummyRegions = [
          { id: 1, name: '문앞' },
          { id: 2, name: '00밭' },
          { id: 3, name: '1번 레인' },
          { id: 4, name: '2번 레인' },
          { id: 5, name: '온실 A동' },
          { id: 6, name: '온실 B동' },
          { id: 7, name: '저장고' },
          { id: 8, name: '뒷마당' },
          { id: 9, name: '주차장' }
        ];

        // API 호출 시뮬레이션
        await new Promise(resolve => setTimeout(resolve, 500));

        setRegions(dummyRegions);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error('구역 데이터 로딩 실패:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchRegions();
  }, []);

  return { regions, loading, error };
}

// 특정 구역 이름으로 구역 찾기
export function useRegionByName(regionName) {
  const { regions, loading, error } = useRegions();
  const [region, setRegion] = useState(null);

  useEffect(() => {
    if (regions.length > 0 && regionName) {
      const foundRegion = regions.find(r => r.name === regionName);
      setRegion(foundRegion || null);
    }
  }, [regions, regionName]);

  return { region, loading, error };
}
