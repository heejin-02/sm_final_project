import axios from 'axios';

const BASE_URL = 'http://localhost:8095';

// 서버 상태 체크
export const checkServerHealth = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/health`, {
      timeout: 3000,
      withCredentials: true
    });
    return true;
  } catch (error) {
    console.warn('서버 연결 상태 확인 실패:', error.message);
    return false;
  }
};

// 일일 통계 데이터 가져오기
export const getDailyStats = async (farmIdx, date) => {
  try {
    const response = await axios.get(`${BASE_URL}/report/daily-stats`, {
      params: {
        farmIdx: farmIdx,
        date: date // YYYY-MM-DD 형식
      },
      withCredentials: true
    });

    return response.data;
  } catch (error) {
    console.error('일일 통계 데이터 조회 실패:', error);
    console.error('에러 상세:', error.response?.data);
    throw error;
  }
};

// 날짜 포맷 유틸리티 함수 (시간대 문제 해결)
export const formatDateForAPI = (date) => {
  if (date instanceof Date) {
    // 로컬 시간대를 유지하면서 YYYY-MM-DD 형식으로 변환
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }
  return date; // 이미 문자열인 경우
};

// 오늘 날짜 가져오기
export const getTodayDate = () => {
  return new Date().toISOString().split('T')[0];
};

// GPT 분석 요약 API
export const getDailyGptSummary = async (farmIdx, date) => {
  try {
    const response = await axios.get('http://192.168.219.72:8000/api/daily-gpt-summary', {
      params: {
        farm_idx: farmIdx,
        date: date
      }
    });
    return response.data;
  } catch (error) {
    console.error('GPT 분석 API 호출 실패:', error);
    throw error;
  }
};

// 일일 구역별 요약 API (백구 메시지용)
export const getDailyZoneSummary = async (farmIdx, date) => {
  try {
    console.log('🚀 getDailyZoneSummary 호출:', { farmIdx, date, getTodayDate });

    const response = await axios.get('http://192.168.219.72:8000/api/daily-gpt-summary', {
      params: {
        farm_idx: farmIdx,
        date: date
      }
    });

    console.log('📥 getDailyZoneSummary 응답:', response.data);
    return response.data;
  } catch (error) {
    console.error('🚨 일일 구역별 요약 API 호출 실패:', error);
    console.error('🚨 에러 상세:', error.response?.data);
    throw error;
  }
};

// 오늘 통계 데이터 가져오기 (today_detecting용)
export const getTodayStats = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/user/today/today`, {
      params: {
        farmIdx: farmIdx
      },
      withCredentials: true,
      timeout: 5000 // 5초 타임아웃
    });
    return response.data;
  } catch (error) {
    console.error('오늘 통계 데이터 조회 실패:', error);
    throw error;
  }
};

// 온실별 오늘 해충 수 가져오기 (farmMap용)
export const getTodayGreenhouses = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/user/today/today/greenhouses`, {
      params: {
        farmIdx: farmIdx
      },
      withCredentials: true,
      timeout: 5000 // 5초 타임아웃
    });
    return response.data;
  } catch (error) {
    console.error('온실별 오늘 해충 수 조회 실패:', error);
    throw error;
  }
};
