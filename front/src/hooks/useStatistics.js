// src/hooks/useStatistics.js
import { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

/**
 * 통계 데이터를 가져오는 커스텀 훅
 * @param {string} period - 'daily', 'monthly', 'yearly'
 * @param {Date} selectedDate - 선택된 날짜
 * @returns {Object} 통계 데이터
 */
export function useStatistics(period, selectedDate = new Date()) {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user?.selectedFarm?.farmIdx || !period) {
      setData(null);
      setLoading(false);
      return;
    }

    // daily는 useDailyStats에서 처리하므로 제외
    if (period === 'daily') {
      setData(null);
      setLoading(false);
      return;
    }

    const fetchStatistics = async () => {
      setLoading(true);
      setError(null);

      try {
        // 날짜 포맷팅
        const formatDate = (date) => {
          const year = date.getFullYear();
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const day = String(date.getDate()).padStart(2, '0');
          return `${year}-${month}-${day}`;
        };

        // 실제 API 호출 (monthly, yearly용)
        const response = await axios.get(`/api/statistics/${period}`, {
          params: {
            farmIdx: user.selectedFarm.farmIdx,
            date: formatDate(selectedDate)
          },
          withCredentials: true
        });

        setData(response.data);
      } catch (err) {
        // console.error('통계 데이터 로딩 실패:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [user?.selectedFarm?.farmIdx, period, selectedDate]);

  return { data, loading, error };
}
