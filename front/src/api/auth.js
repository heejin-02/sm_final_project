// 인증 관련 api
import axios from 'axios';
import { DUMMY_USERS } from '../mocks/users';

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

export async function getUserFarms(userPhone) {
  // 실제 백엔드: return axios.get(`/web/api/user/${userPhone}/farms`);
  const user = DUMMY_USERS.find(u => u.user_phone === userPhone);
  return Promise.resolve({
    data: (user?.farms || []).map(f => ({
      id: f.farm_idx,
      name: f.farm_name,
    })),
  });
}