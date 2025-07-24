import axios from "axios";

// 더미 데이터 정의
const DUMMY_ALERTS = [
  {
    id: 1,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "A구역",
    timestamp: "2025-07-24 14:23:30",
  },
  {
    id: 2,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "A구역",
    timestamp: "2025-07-24 14:23:30",
  },  
  {
    id: 3,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "A구역",
    timestamp: "2025-07-24 14:23:30",
  }, 
  {
    id: 4,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "A구역",
    timestamp: "2025-07-24 14:23:30",
  },   
  {
    id: 5,
    bugName: "알락수염노린재",
    accuracy: 85,
    location: "B구역",
    timestamp: "2025-07-24 13:10:15",
  },
  {
    id: 6,
    bugName: "꽃노랑총채벌레",
    accuracy: 90,
    location: "A구역",
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
