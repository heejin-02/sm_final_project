// 인증 관련 api
import axios from 'axios';
import { DUMMY_USERS } from '../mocks/users';

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
  // DB 연동 코드 잠시 주석
  // api.get('/api/user/farms', { params: { userPhone } });
  const user = DUMMY_USERS.find(u => u.user_phone === userPhone);
  return Promise.resolve({
    data: (user?.farms || []).map(f => ({
      id: f.farm_idx,
      name: f.farm_name,
    })),
  });
}
