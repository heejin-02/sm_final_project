// api/alert.js
import axios from 'axios';
import { API } from './http'; // axios 인스턴스

const BASE_PATH = '/user/alert';

/**
 * 알림 목록 조회 (NotiList에서 사용)
 */
export const getAlertList = async (farmIdx) => {
  try {
    const response = await API.get(`${BASE_PATH}/list/${farmIdx}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * 알림 읽음 처리 후 상세 조회
 */
export const readAndGetAlertDetail = async (anlsIdx) => {
  try {
    const response = await API.get(`${BASE_PATH}/read-and-detail/${anlsIdx}`);
    return response.data;
  } catch (error) {
    console.error('알림 읽음 처리 및 상세 조회 실패:', error);
    throw error;
  }
};

/**
 * 알림 상세 정보 조회
 */
export const getAlertDetail = async (anlsIdx) => {
  try {
    const response = await API.get(`${BASE_PATH}/detail/${anlsIdx}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

// 별칭 export
export const fetchAlertList = getAlertList;
export const fetchAlertDetail = getAlertDetail;
export const markAlertAsRead = readAndGetAlertDetail;
