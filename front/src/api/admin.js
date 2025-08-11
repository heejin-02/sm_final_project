// api/admin.js
import axios from 'axios';
import { API } from './http';

// 전체 회원 목록 조회
export const getAllUsers = () =>
  API.get('/api/admin/users/list');

// 회원 상세 조회
export const getUserDetail = (userPhone) =>
  API.get(`/api/admin/users/${userPhone}`);

// 회원 등록
export const addUser = (userData) =>
  API.post('/admin/user/insert', userData);

// 농장 수정
export const updateFarm = (farmIdx, farmData) =>
  API.put(`/api/admin/users/farms/${farmIdx}`, farmData);

// 농장 삭제
export const deleteFarm = (farmIdx) =>
  API.delete(`/api/admin/users/farm/${farmIdx}`);