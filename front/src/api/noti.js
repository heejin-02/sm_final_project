import axios from "axios";

// 더미 데이터 정의
const DUMMY_ALERTS = [
  {
    id: 1,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "문앞",
    timestamp: "2025-07-24 14:23:30",
  },
  {
    id: 2,
    bugName: "담배가루이",
    accuracy: 85,
    location: "00밭",
    timestamp: "2025-07-24 14:23:30",
  },
  {
    id: 3,
    bugName: "비단노린재",
    accuracy: 88,
    location: "1번 레인",
    timestamp: "2025-07-24 14:23:30",
  },
  {
    id: 4,
    bugName: "꽃노랑총채벌레",
    accuracy: 92,
    location: "온실 A동",
    timestamp: "2025-07-24 14:23:30",
  },
  {
    id: 5,
    bugName: "알락수염노린재",
    accuracy: 85,
    location: "2번 레인",
    timestamp: "2025-07-24 13:10:15",
  },
  {
    id: 6,
    bugName: "담배가루이",
    accuracy: 87,
    location: "온실 B동",
    timestamp: "2025-07-24 14:23:30",
  },
  // …필요한 만큼 더
];

// 더미 반환 함수
export async function fetchUnreadAlerts() {
  // 나중에 실제 axios 호출로 교체!
  return Promise.resolve(DUMMY_ALERTS);
}

// 나중에 real API로 바꿀 부분
// export async function fetchUnreadAlerts() {
//   const res = await axios.get("/api/home/unreadAlerts", {
//     withCredentials: true
//   });
//   return res.data.items;
// }
