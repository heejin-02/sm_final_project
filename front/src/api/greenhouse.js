// src/api/greenhouse.js
// 구역(온실) 관련 API 함수들
import { API } from './http';

const BASE_PATH = '/api/greenhouses';
const TODAY_PATH = '/user/today/today/greenhouses'; // 백엔드 경로가 이렇게 되어 있다면 그대로 둠

// 농장의 구역 목록 조회
export const getGreenhousesByFarm = (farmIdx) =>
  API.get(`${BASE_PATH}/${farmIdx}`, { timeout: 5000 }).then(res => res.data);

// 구역 정보 생성/수정 (배열 전송)
export const saveGreenhouses = (farmIdx, greenhouseData) =>
  API.post(`${BASE_PATH}/${farmIdx}`, greenhouseData, { timeout: 10000 }).then(res => res.data);

// 특정 구역 정보 조회
export const getGreenhouseDetail = (farmIdx, ghIdx) =>
  API.get(`${BASE_PATH}/${farmIdx}/${ghIdx}`, { timeout: 5000 }).then(res => res.data);

// 구역 삭제
export const deleteGreenhouse = (farmIdx, ghIdx) =>
  API.delete(`${BASE_PATH}/${farmIdx}/${ghIdx}`, { timeout: 5000 }).then(res => res.data);

// 구역별 오늘 해충 탐지 수 조회
export const getTodayGreenhouseStats = (farmIdx) =>
  API.get(TODAY_PATH, { params: { farmIdx }, timeout: 5000 }).then(res => res.data);

// 구역 이름 업데이트
export const updateGreenhouseName = (farmIdx, ghIdx, ghName) =>
  API.put(`${BASE_PATH}/${farmIdx}/${ghIdx}/name`, { ghName }, { timeout: 5000 }).then(res => res.data);
