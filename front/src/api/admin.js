// api/admin.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8095',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' }
});

// 전체 회원 목록 조회
export const getAllUsers = () =>
  api.get('/api/admin/users/list');

// 회원 상세 조회
export const getUserDetail = (userPhone) =>
  api.get(`/api/admin/users/${userPhone}`);

// 회원 등록
export const addUser = (userData) =>
  api.post('/admin/user/insert', userData);

// 농장 수정
export const updateFarm = (farmIdx, farmData) =>
  api.put(`/api/admin/users/farms/${farmIdx}`, farmData);

// 농장 삭제
export const deleteFarm = (farmIdx) =>
  api.delete(`/api/admin/users/farm/${farmIdx}`);