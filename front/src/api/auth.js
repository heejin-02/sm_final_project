// 인증 관련 api
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8095/web/api',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
});

// 로그인
export const login = (userPhone, userPw) =>
  api.post('/auth/login', { userPhone, userPw });

// 로그아웃 요청
export const logout = () =>
  api.post('/auth/logout');

// 백엔드에서 
export const getCurrentUser = () =>
  api.get('/auth/me');

// 로그인한 사용자의 농장정보 받아오기
export const getFarmInfo = () =>
  api.get('/auth/farmInfo');