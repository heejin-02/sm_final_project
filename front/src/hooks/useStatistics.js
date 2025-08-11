import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  getDailyStats,
  getMonthlyStats,
  getYearlyStats,
  formatDateForAPI,
  formatMonthForAPI,
  formatYearForAPI
} from '../api/report';

export function useStatistics({ period, date }) {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    if (!period || !date || !user?.selectedFarm?.farmIdx) {
      setStats(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let formatted;
      let fetchFn;

      switch (period) {
        case 'daily':
          formatted = formatDateForAPI(date);
          fetchFn = () => getDailyStats(user.selectedFarm.farmIdx, formatted);
          break;
        case 'monthly':
          formatted = formatMonthForAPI(date);
          fetchFn = () => getMonthlyStats(user.selectedFarm.farmIdx, formatted);
          break;
        case 'yearly':
          formatted = formatYearForAPI(date);
          fetchFn = () => getYearlyStats(user.selectedFarm.farmIdx, formatted);
          break;
        default:
          throw new Error('지원하지 않는 기간입니다.');
      }

      const data = await fetchFn();
      setStats(data);
    } catch (e) {
      setError(e.message || '통계 데이터를 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  }, [period, date, user]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    stats,
    loading,
    error,
    refetch: fetchData
  };
}
