// 서버에서 구역별 해충 카운트(fetch) 담당
import axios from "axios";
import { getRegionsWithDetectionCount } from "../mock/greenHouse";

// 나중에 real API endpoint로만 바꿔주세요
export async function fetchRegionCounts(farmId) {
  // return axios
  //   .get(`/api/home/farms/${farmId}/region-counts`, { withCredentials:true })
  //   .then(res => res.data);

  // mock 데이터 사용 (DB + 기본값 병합)
  const regions = getRegionsWithDetectionCount(parseInt(farmId));

  // API 호출 시뮬레이션을 위한 딜레이
  await new Promise(resolve => setTimeout(resolve, 300));

  // 9개 구역 보장 (1~9번)
  const result = [];
  for (let i = 1; i <= 9; i++) {
    const existingRegion = regions.find(r => r.ghIdx === i);
    if (existingRegion) {
      // 실제 데이터가 있으면 사용
      result.push({
        id: existingRegion.ghIdx,
        name: existingRegion.gh_name,
        count: existingRegion.count || 0
      });
    } else {
      // 없으면 기본 구역 생성
      result.push({
        id: i,
        name: `${i}번 구역`,
        count: Math.floor(Math.random() * 3) // 0~2 랜덤 더미 카운트
      });
    }
  }

  return result;
}
