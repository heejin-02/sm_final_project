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
