// 서버에서 구역별 해충 카운트(fetch) 담당
import axios from "axios";

// 나중에 real API endpoint로만 바꿔주세요
export async function fetchRegionCounts(farmId) {
  // return axios
  //   .get(`/api/home/farms/${farmId}/region-counts`, { withCredentials:true })
  //   .then(res => res.data);
  
  // 더미
  return Promise.resolve([
    { cctv_idx: 1, gh_idx: 1, count: 5 },
    { cctv_idx: 2, gh_idx: 1, count: 10 },
    { cctv_idx: 3, gh_idx: 1, count: 15 },
    { cctv_idx: 4, gh_idx: 1, count: 20 },
    { cctv_idx: 5, gh_idx: 1, count: 25 },	
		{ cctv_idx: 6, gh_idx: 1, count: 30 },
		{ cctv_idx: 7, gh_idx: 1, count: 35 },
		{ cctv_idx: 8, gh_idx: 1, count: 40 },
		{ cctv_idx: 9, gh_idx: 1, count: 45 }
  ]);
}
