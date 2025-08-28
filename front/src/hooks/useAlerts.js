// hooks/useAlerts.js
// 알림 관련 커스텀 훅들

import { useState, useEffect } from 'react';
import { fetchAlertList, fetchAlertDetail, markAlertAsRead } from '../api/alert';

/**
 * 알림 목록을 관리하는 훅 (NotiList에서 사용)
 * @param {number} farmIdx - 농장 인덱스
 * @returns {Object} { alerts, loading, error, refreshAlerts }
 */
export const useAlertList = (farmIdx) => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadAlerts = async () => {
    if (!farmIdx) return;

    try {
      setLoading(true);
      setError(null);
      const data = await fetchAlertList(farmIdx);
      setAlerts(data || []);
    } catch (err) {
      // console.error('알림 목록 로딩 실패:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAlerts();
  }, [farmIdx]);

  // 알림 목록 새로고침
  const refreshAlerts = () => {
    loadAlerts();
  };

  // 알림 읽음 처리
  const markAsRead = async (anlsIdx) => {
    try {
      await markAlertAsRead(anlsIdx);
      // 로컬 상태 업데이트 (notiCheck를 Y로 변경)
      setAlerts(prev => prev.map(alert =>
        alert.anlsIdx === anlsIdx
          ? { ...alert, notiCheck: "Y" }
          : alert
      ));
    } catch (err) {
      console.error('알림 읽음 처리 실패:', err);
    }
  };

  return {
    alerts,
    loading,
    error,
    refreshAlerts,
    markAsRead
  };
};

/**
 * 알림 상세 정보를 관리하는 훅 (NotiDetail에서 사용)
 * @param {number} anlsIdx - 분석 인덱스
 * @returns {Object} { alertDetail, loading, error, refreshDetail }
 */
export const useAlertDetail = (anlsIdx) => {
  const [alertDetail, setAlertDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadAlertDetail = async () => {
    if (!anlsIdx) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await fetchAlertDetail(anlsIdx);
      setAlertDetail(data);
    } catch (err) {
      // console.error('알림 상세 정보 로딩 실패:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAlertDetail();
  }, [anlsIdx]);

  // 상세 정보 새로고침
  const refreshDetail = () => {
    loadAlertDetail();
  };

  return {
    alertDetail,
    loading,
    error,
    refreshDetail
  };
};

/**
 * 읽지 않은 알림 개수를 관리하는 훅
 * @param {number} farmIdx - 농장 인덱스
 * @returns {Object} { unreadCount, loading }
 */
export const useUnreadAlertCount = (farmIdx) => {
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadUnreadCount = async () => {
      if (!farmIdx) return;
      
      try {
        setLoading(true);
        const alerts = await fetchAlertList(farmIdx);
        const count = alerts.filter(alert => alert.notiCheck !== "Y").length;
        setUnreadCount(count);
      } catch (err) {
        // console.error('읽지 않은 알림 개수 로딩 실패:', err);
      } finally {
        setLoading(false);
      }
    };

    loadUnreadCount();
  }, [farmIdx]);

  return { unreadCount, loading };
};
