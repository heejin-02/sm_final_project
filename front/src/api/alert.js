// api/alert.js
// 알림 관련 API 함수들

import axios from 'axios';

const BASE_URL = 'http://localhost:8095/user/alert';

/**
 * 알림 목록 조회 (NotiList에서 사용)
 * @param {number} farmIdx - 농장 인덱스
 * @returns {Promise<Array>} 알림 목록
 */
export const getAlertList = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/list/${farmIdx}`, {
      withCredentials: true
    });
    return response.data;
  } catch (error) {
    console.error('알림 목록 조회 실패:', error);
    throw error;
  }
};

/**
 * 알림 읽음 처리 후 상세 조회 (NotiList에서 알림 클릭 시 사용)
 * @param {number} anlsIdx - 분석 인덱스
 * @returns {Promise<Object>} 알림 상세 정보
 */
export const readAndGetAlertDetail = async (anlsIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/read-and-detail/${anlsIdx}`, {
      withCredentials: true
    });
    return response.data;
  } catch (error) {
    console.error('알림 읽음 처리 및 상세 조회 실패:', error);
    throw error;
  }
};

/**
 * 알림 상세 정보 조회 (NotiDetail에서 사용)
 * 구역 확인, 동영상 확인, GPT 확인 등 전체 상세 정보
 * @param {number} anlsIdx - 분석 인덱스
 * @returns {Promise<Object>} 알림 전체 상세 정보
 */
export const getAlertDetail = async (anlsIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/detail/${anlsIdx}`, {
      withCredentials: true
    });
    return response.data;
  } catch (error) {
    console.error('알림 상세 정보 조회 실패:', error);
    throw error;
  }
};

// 개발용 더미 데이터 (실제 백엔드 API 구조에 맞춤)
const DUMMY_ALERTS = [
  {
    anlsIdx: 999999991, // DB 시퀀스와 절대 안겹치는 큰 수
    insectName: "알락수염노린재",
    ghArea: "100㎡",
    anlsAcc: 85,
    createdAt: "2025-01-30 14:30:00",
    notiCheck: null,
    userQes: "gpt 응답",
    gptContent: "알락수염노린재가 탐지되었습니다. 즉시 방제 조치가 필요합니다.",
    isRead: false // 프론트엔드용 추가 필드
  },
  {
    anlsIdx: 999999992,
    insectName: "꽃노랑총채벌레",
    ghArea: "100㎡",
    anlsAcc: 92,
    createdAt: "2025-01-30 13:15:00",
    notiCheck: null,
    userQes: "gpt 응답",
    gptContent: "꽃노랑총채벌레가 탐지되었습니다. 주의 깊은 관찰이 필요합니다.",
    isRead: false
  },
  {
    anlsIdx: 999999993,
    insectName: "담배가루이",
    ghArea: "100㎡",
    anlsAcc: 78,
    createdAt: "2025-01-30 11:45:00",
    notiCheck: null,
    userQes: "gpt 응답",
    gptContent: "담배가루이가 탐지되었습니다. 환경 관리가 필요합니다.",
    isRead: true
  }
];

const DUMMY_ALERT_DETAIL = {
  greenhouseInfo: {
    ghArea: "100㎡",
    createdAt: "2025-01-30 14:30:00",
    anlsAcc: "85",
    insectName: "알락수염노린재"
  },
  imageList: [
    {
      imgUrl: "/20250730/dummy_20250730_143000.mp4",
      createdAt: "2025-01-30 14:30:00",
      anlsIdx: 999999991
    }
  ],
  gptResult: {
    gptIdx: 999999991,
    userQes: "gpt 응답",
    userFile1: null,
    userFile2: null,
    userFile3: null,
    gptContent: "알락수염노린재는 주로 식물의 즙을 빨아먹으며, 그로 인해 작물의 성장이 저해될 수 있어요. 이 해충은 매우 빠르게 번식하니 온실에서 자주 확인해 보셔야 합니다. 방제 방법으로는 해충이 발견된 식물에 예방용 농약을 뿌리거나, 손으로 직접 제거하는 것도 좋은 방법이랍니다.",
    createdAt: "2025-01-30 14:30:00",
    anlsIdx: null
  }
};

/**
 * 개발용 더미 데이터 함수들
 */
export const getDummyAlertList = async (farmIdx) => {
  // API 호출 시뮬레이션
  await new Promise(resolve => setTimeout(resolve, 500));
  // 실제 API는 farmIdx로 필터링하지만, 더미에서는 모든 데이터 반환
  return DUMMY_ALERTS;
};

export const getDummyAlertDetail = async (anlsIdx) => {
  // API 호출 시뮬레이션
  await new Promise(resolve => setTimeout(resolve, 300));
  return { ...DUMMY_ALERT_DETAIL, anlsIdx };
};

export const markDummyAsRead = async (anlsIdx) => {
  // 읽음 처리 시뮬레이션
  await new Promise(resolve => setTimeout(resolve, 200));
  const alert = DUMMY_ALERTS.find(a => a.anlsIdx === anlsIdx);
  if (alert) {
    alert.isRead = true;
  }
  return alert;
};

// 개발 모드 여부에 따라 실제 API 또는 더미 데이터 사용
const isDevelopment = true; // true: 더미 데이터 사용, false: 실제 API 사용 (일단 테스트용)



export const fetchAlertList = isDevelopment ? getDummyAlertList : getAlertList;
export const fetchAlertDetail = isDevelopment ? getDummyAlertDetail : getAlertDetail;
export const markAlertAsRead = isDevelopment ? markDummyAsRead : readAndGetAlertDetail;
