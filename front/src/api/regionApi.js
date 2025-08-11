// 서버에서 구역별 해충 카운트(fetch) 담당
import { API } from './http'; // axios 인스턴스

// 구역별 해충 카운트 조회
export async function fetchRegionCounts(farmId) {
  try {
    const response = await API.get('/user/today/today/greenhouses', {
      params: { farmIdx: farmId },
      timeout: 5000
    });

    const apiData = response.data || [];

    // 9개 구역 보장 (1~9번)
    const result = [];
    for (let i = 1; i <= 9; i++) {
      const existingRegion = apiData.find(r => r.ghIdx === i);
      if (existingRegion) {
        result.push({
          id: existingRegion.ghIdx,
          name: existingRegion.ghName || `${i}번 구역`,
          count: existingRegion.todayInsectCount || 0
        });
      } else {
        result.push({
          id: i,
          name: `${i}번 구역`,
          count: 0
        });
      }
    }

    return result;
  } catch (error) {
    console.error('구역별 해충 카운트 조회 실패:', error);

    // API 실패 시 기본 구역 데이터 반환
    return Array.from({ length: 9 }, (_, idx) => ({
      id: idx + 1,
      name: `${idx + 1}번 구역`,
      count: 0
    }));
  }
}
