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

  return regions.map(region => ({
    farm_idx: region.farmIdx,
    gh_idx: region.ghIdx,
    gh_name: region.gh_name,
    count: region.count
  }));
}
