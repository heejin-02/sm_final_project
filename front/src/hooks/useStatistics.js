// hooks/useStatistics.js
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

  // (선택) 이전 요청 취소용
  const abortRef = useRef(null);

  const fetchData = useCallback(async () => {
    // 기본 가드
    if (!period || !date || !user?.selectedFarm?.farmIdx) {
      setStats(null);
      return;
    }

    // 이전 요청 취소
    abortRef.current?.abort?.();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);

    let formatted = null;
    let fetchFn = null;

    try {
      // period에 맞게 포맷/함수 세팅
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
        default: {
          throw new Error(`지원하지 않는 기간입니다: ${period}`);
        }
      }

      // fetchFn이 없으면 즉시 중단 (타이머도 종료)
      if (!fetchFn) {
        throw new Error('내부 설정 오류: fetchFn 미할당');
      }

      const timerLabel = `stats-${period}`;
      console.time(timerLabel);
      const data = await fetchFn({ signal: abortRef.current.signal }); // axios면 무시됨, fetch면 전달됨
      console.timeEnd(timerLabel);

      setStats(data);
    } catch (e) {
      // 취소는 조용히 무시
      if (e?.name === 'CanceledError' || e?.code === 'ERR_CANCELED' || e?.name === 'AbortError') {
        return;
      }
      setError(e?.message || '통계 데이터를 불러오는데 실패했습니다.');
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, [period, date, user]);

  useEffect(() => {
    fetchData();
    // 언마운트 시 in-flight 취소
    return () => abortRef.current?.abort?.();
  }, [fetchData]);

  return { stats, loading, error, refetch: fetchData };
}
