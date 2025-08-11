import { useState, useEffect, useCallback, useRef } from 'react';
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

  // 겹치는 호출 식별용
  const callIdRef = useRef(0);

  const fetchData = useCallback(async () => {
    if (!period || !date || !user?.selectedFarm?.farmIdx) {
      setStats(null);
      return;
    }

    setLoading(true);
    setError(null);

    // 이번 호출 id
    const myId = ++callIdRef.current;

    try {
      let formatted;
      let fetchFn;

      switch (period) {
        case 'daily': {
          formatted = formatDateForAPI(date);
          fetchFn = () => getDailyStats(user.selectedFarm.farmIdx, formatted);
          break;
        }
        case 'monthly': {
          formatted = formatMonthForAPI(date);
          fetchFn = () => getMonthlyStats(user.selectedFarm.farmIdx, formatted);
          break;
        }
        case 'yearly': {
          formatted = formatYearForAPI(date);
          fetchFn = () => getYearlyStats(user.selectedFarm.farmIdx, formatted);
          break;
        }
        default:
          throw new Error(`지원하지 않는 기간입니다: ${period}`);
      }

      const label = `stats-${period}-${myId}`; // 라벨 유니크

      const data = await fetchFn();            // axios timeout 10s면 여기서 취소될 수 있음
          

      // 더 최신 호출이 있으면 내 결과는 무시
      if (myId !== callIdRef.current) return;

      setStats(data);
    } catch (e) {
      if (myId !== callIdRef.current) return; // 최신 호출만 에러 반영
      setError(e?.message || '통계 데이터를 불러오는데 실패했습니다.');
      setStats(null);
    } finally {
      if (myId === callIdRef.current) setLoading(false);
    }
  }, [period, date, user?.selectedFarm?.farmIdx]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { stats, loading, error, refetch: fetchData };
}
