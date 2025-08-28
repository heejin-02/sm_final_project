// 서버에서 구역별 해충 카운트(fetch) 담당
import axios from "axios";

// 구역별 해충 카운트 조회 (실제 API 연동)
export async function fetchRegionCounts(farmId) {
  try {
    // 실제 API 호출 (getTodayGreenhouses와 동일한 엔드포인트 사용)
    const response = await axios.get(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8095'}/user/today/today/greenhouses`, {
      params: { 
        farmIdx: farmId,
        _t: Date.now()
      },
      withCredentials: true,
      timeout: 3000
    });

    const apiData = response.data || [];
    // console.log('getTodayGreenhouses API 응답:', apiData);

    // 9개 구역 보장 (1~9번)
    const result = [];
    for (let i = 1; i <= 9; i++) {
      const existingRegion = apiData.find(r => r.ghIdx === i);
      if (existingRegion) {
        // API에서 받은 실제 데이터 사용
        result.push({
          id: existingRegion.ghIdx,
          name: existingRegion.ghName || `${i}번 구역`,
          count: existingRegion.todayInsectCount || 0
        });
      } else {
        // API에 없는 구역은 기본값 사용
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
    const fallbackResult = [];
    for (let i = 1; i <= 9; i++) {
      fallbackResult.push({
        id: i,
        name: `${i}번 구역`,
        count: 0
      });
    }
    return fallbackResult;
  }
}
