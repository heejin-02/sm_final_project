// src/hooks/useRegions.js
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { getTodayGreenhouses } from '../api/report';

export function useRegions() {
  const [regions, setRegions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useAuth();
  const farmIdx = user?.selectedFarm?.farmIdx;

  useEffect(() => {
    const fetchRegions = async () => {
      if (!farmIdx) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);

        // 실제 API 호출 (getTodayGreenhouses 직접 사용)
        const apiData = await getTodayGreenhouses(farmIdx);
        // API 데이터를 regions 형태로 변환 (실제 데이터만 사용)
        const regionsWithNames = apiData.map(region => ({
          id: region.ghIdx,
          name: region.ghName || `${region.ghIdx}번 구역`,
          count: region.todayInsectCount || 0
        }));
        setRegions(regionsWithNames);
        setError(null);
      } catch (err) {
        console.error('구역 데이터 로딩 실패:', err);
        setError(err.message);

        // API 실패 시 기본 구역 데이터 사용
        const fallbackRegions = [];
        for (let i = 1; i <= 9; i++) {
          fallbackRegions.push({
            id: i,
            name: `${i}번 구역`,
            count: 0
          });
        }
        setRegions(fallbackRegions);
      } finally {
        setLoading(false);
      }
    };

    fetchRegions();
  }, [farmIdx]);

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
