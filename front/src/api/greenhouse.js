// src/api/greenhouse.js
// 구역(온실) 관련 API 함수들

import axios from 'axios';

const BASE_URL = 'smfinalproject-production-88a2.up.railway.app';

// 농장의 구역 목록 조회
export const getGreenhousesByFarm = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/api/greenhouses/${farmIdx}`, {
      withCredentials: true,
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    console.error('구역 목록 조회 실패:', error);
    throw error;
  }
};

// 구역 정보 생성/수정
export const saveGreenhouses = async (farmIdx, greenhouseData) => {
  try {
    const response = await axios.post(`${BASE_URL}/api/greenhouses/${farmIdx}`, greenhouseData, {
      withCredentials: true,
      timeout: 10000
    });
    return response.data;
  } catch (error) {
    console.error('구역 정보 저장 실패:', error);
    throw error;
  }
};

// 특정 구역 정보 조회
export const getGreenhouseDetail = async (farmIdx, ghIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/api/greenhouses/${farmIdx}/${ghIdx}`, {
      withCredentials: true,
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    console.error('구역 상세 정보 조회 실패:', error);
    throw error;
  }
};

// 구역 삭제
export const deleteGreenhouse = async (farmIdx, ghIdx) => {
  try {
    const response = await axios.delete(`${BASE_URL}/api/greenhouses/${farmIdx}/${ghIdx}`, {
      withCredentials: true,
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    console.error('구역 삭제 실패:', error);
    throw error;
  }
};

// 구역별 오늘 해충 탐지 수 조회 (기존 getTodayGreenhouses와 동일)
export const getTodayGreenhouseStats = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/user/today/today/greenhouses`, {
      params: { farmIdx },
      withCredentials: true,
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    console.error('구역별 오늘 통계 조회 실패:', error);
    throw error;
  }
};

// 구역 이름 업데이트 (단일 구역)
export const updateGreenhouseName = async (farmIdx, ghIdx, ghName) => {
  try {
    const response = await axios.put(`${BASE_URL}/api/greenhouses/${farmIdx}/${ghIdx}/name`, {
      ghName
    }, {
      withCredentials: true,
      timeout: 5000
    });
    return response.data;
  } catch (error) {
    console.error('구역 이름 업데이트 실패:', error);
    throw error;
  }
};
