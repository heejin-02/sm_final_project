import { API, GPT_API } from './http';

// 서버 상태 체크
export const checkServerHealth = async () => {
  try {
    await API.get('/health', { timeout: 3000 });
    return true;
  } catch (error) {
    console.warn('서버 연결 상태 확인 실패:', error.message);
    return false;
  }
};

// 일일 통계 데이터
export const getDailyStats = (farmIdx, date) =>
  API.get('/report/daily-stats', { params: { farmIdx, date } }).then(res => res.data);

// GPT 분석 요약
export const getDailyGptSummary = (farmIdx, date) =>
  GPT_API.get('/api/daily-gpt-summary', { params: { farm_idx: farmIdx, date } }).then(res => res.data);

export const getMonthlyGptSummary = (farmIdx, date) =>
  GPT_API.get('/api/monthly-gpt-summary', { params: { farm_idx: farmIdx, month: date } }).then(res => res.data);

export const getYearlyGptSummary = (farmIdx, date) =>
  GPT_API.get('/api/yearly-gpt-summary', { params: { farm_idx: farmIdx, year: date } }).then(res => res.data);

// 일일 구역별 요약 (백구 메시지용)
export const getDailyZoneSummary = (farmIdx, date) =>
  GPT_API.get('/api/daily-gpt-summary', { params: { farm_idx: farmIdx, date } }).then(res => res.data);

// 오늘 통계 데이터
export const getTodayStats = (farmIdx) =>
  API.get('/user/today/today', { params: { farmIdx }, timeout: 5000 }).then(res => res.data);

// 온실별 오늘 해충 수
export const getTodayGreenhouses = (farmIdx) =>
  API.get('/user/today/today/greenhouses', { params: { farmIdx }, timeout: 3000 }).then(res => res.data);

// 월간 통계 데이터
export const getMonthlyStats = (farmIdx, month) =>
  API.get('/report/monthly-stats', { params: { farmIdx, month }, timeout: 10000 }).then(res => res.data);

// 연간 통계 데이터
export const getYearlyStats = (farmIdx, year) =>
  API.get('/report/yearly-stats', { params: { farmIdx, year }, timeout: 10000 }).then(res => res.data);

// 날짜 포맷 유틸
export const formatDateForAPI = (date) =>
  date instanceof Date
    ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
    : date;

export const getTodayDate = () => new Date().toISOString().split('T')[0];

export const formatMonthForAPI = (date) =>
  date instanceof Date
    ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
    : date;

export const formatYearForAPI = (date) =>
  date instanceof Date ? `${date.getFullYear()}` : date;
