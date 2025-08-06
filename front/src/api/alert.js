// api/alert.js
// 알림 관련 API 함수들

import axios from 'axios';

const BASE_URL = 'http://localhost:8095/user/alert';

/**
 * 알림 목록 조회 (NotiList에서 사용)
 * @param {number} farmIdx - 농장 인덱스
 * @returns {Promise<Array>} 알림 목록
 */
export const getAlertList = async (farmIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/list/${farmIdx}`, {
      withCredentials: true
    });
    return response.data;
  } catch (error) {
    // console.error('알림 목록 조회 실패:', error);
    throw error;
  }
};

/**
 * 알림 읽음 처리 후 상세 조회 (NotiList에서 알림 클릭 시 사용)
 * @param {number} anlsIdx - 분석 인덱스
 * @returns {Promise<Object>} 알림 상세 정보
 */
export const readAndGetAlertDetail = async (anlsIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/read-and-detail/${anlsIdx}`, {
      withCredentials: true
    });
    return response.data;
  } catch (error) {
    console.error('알림 읽음 처리 및 상세 조회 실패:', error);
    throw error;
  }
};

/**
 * 알림 상세 정보 조회 (NotiDetail에서 사용)
 * 구역 확인, 동영상 확인, GPT 확인 등 전체 상세 정보
 * @param {number} anlsIdx - 분석 인덱스
 * @returns {Promise<Object>} 알림 전체 상세 정보
 */
export const getAlertDetail = async (anlsIdx) => {
  try {
    const response = await axios.get(`${BASE_URL}/detail/${anlsIdx}`, {
      withCredentials: true
    });
    // console.log('getAlertDetail API 응답:', response.data);
    return response.data;
  } catch (error) {
    // console.error('알림 상세 정보 조회 실패:', error);
    throw error;
  }
};

// 실제 API 함수들을 직접 export
export const fetchAlertList = getAlertList;
export const fetchAlertDetail = getAlertDetail;
export const markAlertAsRead = readAndGetAlertDetail;
