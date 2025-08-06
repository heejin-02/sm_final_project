// src/hooks/useStatistics.js
import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  getDailyStats,
  getMonthlyStats,
  getYearlyStats,
  getDailyGptSummary,
  getMonthlyGptSummary,
  // getYearlyGptSummary,
  formatDateForAPI,
  formatMonthForAPI,
  formatYearForAPI
} from '../api/report';

export function useStatistics({ period, date }) {
  const { user } = useAuth();

  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [gptSummary, setGptSummary] = useState(null);
  const [gptLoading, setGptLoading] = useState(false);
  const [gptError, setGptError] = useState(null);

  const fetchData = useCallback(async () => {
    if (!period || !date || !user?.selectedFarm?.farmIdx) {
      setStats(null);
      return;
    }

    setLoading(true);
    setError(null);

    let formatted;
    let fetchStatsFn;
    let fetchGptFn;

    switch (period) {
      case 'daily':
        formatted = formatDateForAPI(date);
        fetchStatsFn = () => getDailyStats(user.selectedFarm.farmIdx, formatted);
        fetchGptFn   = () => getDailyGptSummary(user.selectedFarm.farmIdx, formatted);
        break;
      case 'monthly':
        formatted = formatMonthForAPI(date);
        fetchStatsFn = () => getMonthlyStats(user.selectedFarm.farmIdx, formatted);
        fetchGptFn   = () => getMonthlyGptSummary(user.selectedFarm.farmIdx, formatted);
        break;
      case 'yearly':
        formatted = formatYearForAPI(date);
        fetchStatsFn = () => getYearlyStats(user.selectedFarm.farmIdx, formatted);
        // fetchGptFn   = () => getYearlyGptSummary(user.selectedFarm.farmIdx, formatted);
        fetchGptFn   = () => '2';        
        break;
      default:
        setError('지원하지 않는 기간입니다.');
        setLoading(false);
        return;
    }

    try {
      const data = await fetchStatsFn();
      setStats(data);

      // GPT 요약
      setGptLoading(true);
      setGptError(null);
      try {
        const summaryData = await fetchGptFn();
        setGptSummary(summaryData.summary);
      } catch (e) {
        console.error('GPT summary error', e);
        setGptError(e.message || 'GPT 분석을 불러오는데 실패했습니다.');
        setGptSummary('분석 중 오류가 발생했습니다.');
      } finally {
        setGptLoading(false);
      }

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
    gptSummary,
    gptLoading,
    gptError,
    refetch: fetchData
  };
}
