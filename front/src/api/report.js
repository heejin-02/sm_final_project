import axios from 'axios';

const BASE_URL = 'http://localhost:8095';

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
    console.error('❌ 일일 통계 데이터 조회 실패:', error);
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
    const response = await axios.get('http://192.168.219.72:8000/api/daily-zone-summary', {
      params: {
        farm_idx: farmIdx,
        date: date
      }
    });
    return response.data;
  } catch (error) {
    console.error('일일 구역별 요약 API 호출 실패:', error);
    throw error;
  }
};
