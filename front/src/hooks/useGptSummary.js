// src/hooks/useGptSummary.js
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  getDailyGptSummary,
  getMonthlyGptSummary,
  getYearlyGptSummary,
  formatDateForAPI,
  formatMonthForAPI,
  formatYearForAPI
} from '../api/report';

export function useGptSummary({ period, date }) {
  const { user } = useAuth();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user?.selectedFarm?.farmIdx || !period || !date) return;

    const fetchSummary = async () => {
      setLoading(true);
      setError(null);

      try {
        let formattedDate;
        let fetchFn;

        if (period === 'daily') {
          formattedDate = formatDateForAPI(date);
          fetchFn = () => getDailyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else if (period === 'monthly') {
          formattedDate = formatMonthForAPI(date);
          fetchFn = () => getMonthlyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else if (period === 'yearly') {
          formattedDate = formatYearForAPI(date);
          fetchFn = () => getYearlyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else {
          throw new Error('지원하지 않는 기간입니다.');
        }

        const result = await fetchFn();
        setSummary(result.summary);
      } catch (err) {
        setError(err.message || 'GPT 요약 실패');
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
  }, [user?.selectedFarm?.farmIdx, period, date]);

  return { gptSummary: summary, gptLoading: loading, gptError: error };
}
