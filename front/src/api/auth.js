// 인증 관련 api
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8095',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' }
});

// 로그인
export const loginCheck = (id, pw) =>
  api.post('/api/home/loginCheck', null, { params: { id, pw } });

// 사용자 - 농장 리스트 조회
export const getUserFarms = (userPhone) => {
  return api.get('/api/user/farms', { params: { userPhone } });
}

// 세션 확인
export const checkSession = async () => {
  try {
    const response = await api.get('/api/home/check-session');
    return response.data;
  } catch (error) {
    console.error('세션 확인 실패:', error);
    // 네트워크 오류 시 로그아웃 상태로 처리
    return {
      isAuthenticated: false,
      user: null
    };
  }
};

// 농장 선택 정보를 세션에 저장
export const selectFarmInSession = async (farmInfo) => {
  try {
    const response = await api.post('/api/home/select-farm', farmInfo);
    return response.data;
  } catch (error) {
    console.error('농장 선택 세션 저장 실패:', error);
    throw error;
  }
};

// 로그아웃
export const logout = async () => {
  try {
    const response = await api.post('/api/home/logout');
    return response.data;
  } catch (error) {
    throw error;
  }
};
