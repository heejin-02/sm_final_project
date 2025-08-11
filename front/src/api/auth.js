// 인증 관련 api
import axios from 'axios';
import { API } from './http';

// 로그인
export const loginCheck = (id, pw) => {
  return API.post('/api/home/loginCheck', { id, pw });
};

// 사용자 - 농장 리스트 조회
export const getUserFarms = (userPhone) => {
  return API.get('/api/user/farms', { params: { userPhone } });
};
// 세션 확인
export const checkSession = async () => {
  try {
    const response = await API.get('/api/home/check-session');
    return response.data;
  } catch (error) {
    console.error('세션 확인 실패:', error);
    return {
      isAuthenticated: false,
      user: null
    };
  }
};

// 로그아웃
export const logout = async () => {
  try {
    const response = await API.post('/api/home/logout');
    return response.data;
  } catch (error) {
    throw error;
  }
};

