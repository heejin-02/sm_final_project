// src/hooks/useGptSummary.js
import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  getDailyGptSummary,
  getMonthlyGptSummary,
  getYearlyGptSummary,
  formatDateForAPI,
  formatMonthForAPI,
  formatYearForAPI,
} from '../api/report';

// 간단한 메모리 캐시
const cache = new Map();
const CACHE_TTL = 60000; // 1분

export function useGptSummary({ period, date }) {
  const { user } = useAuth();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    if (!user?.selectedFarm?.farmIdx || !period || !date) return;

    // 캐시 키 생성
    const cacheKey = `${user.selectedFarm.farmIdx}-${period}-${date}`;

    // 캐시 확인
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      setSummary(cached.data);
      return;
    }

    const fetchSummary = async () => {
      // 이전 요청 취소
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // 새 컨트롤러 생성
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setLoading(true);
      setError(null);

      try {
        let formattedDate;
        let fetchFn;

        if (period === 'daily') {
          formattedDate = formatDateForAPI(date);
          fetchFn = () =>
            getDailyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else if (period === 'monthly') {
          formattedDate = formatMonthForAPI(date);
          fetchFn = () =>
            getMonthlyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else if (period === 'yearly') {
          formattedDate = formatYearForAPI(date);
          fetchFn = () =>
            getYearlyGptSummary(user.selectedFarm.farmIdx, formattedDate);
        } else {
          throw new Error('지원하지 않는 기간입니다.');
        }

        const result = await fetchFn();

        // 취소되지 않았을 때만 상태 업데이트
        if (!controller.signal.aborted) {
          const summaryData = result.summary;
          setSummary(summaryData);

          // 캐시에 저장
          cache.set(cacheKey, {
            data: summaryData,
            timestamp: Date.now(),
          });
        }
      } catch (err) {
        if (!abortControllerRef.current?.signal.aborted) {
          setError(err.message || 'GPT 요약 실패');
          console.error('GPT 요약 실패:', err);
        }
      } finally {
        if (!abortControllerRef.current?.signal.aborted) {
          setLoading(false);
        }
      }
    };

    fetchSummary();

    // Cleanup
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [user?.selectedFarm?.farmIdx, period, date]);

  return { gptSummary: summary, gptLoading: loading, gptError: error };
}
