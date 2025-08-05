import { useState, useEffect } from 'react';
import { getDailyStats, getDailyGptSummary, formatDateForAPI } from '../api/report';
import { useAuth } from '../contexts/AuthContext';

export const useDailyStats = (selectedDate = null) => {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // GPT 분석 상태
  const [gptSummary, setGptSummary] = useState(null);
  const [gptLoading, setGptLoading] = useState(false);
  const [gptError, setGptError] = useState(null);

  const fetchDailyStats = async (date = null) => {
    if (!user?.selectedFarm?.farmIdx) {
      setError('농장이 선택되지 않았습니다.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // 전달받은 날짜를 그대로 사용 (하루 빼기 없음)
      const targetDate = date || selectedDate;
      if (!targetDate) {
        setError('날짜가 지정되지 않았습니다.');
        return;
      }

      const formattedDate = formatDateForAPI(targetDate);

      const data = await getDailyStats(user.selectedFarm.farmIdx, formattedDate);
      setStats(data);

      // 통계 데이터 로드 후 GPT 분석 요청
      fetchGptSummary(user.selectedFarm.farmIdx, formattedDate);
    } catch (err) {
      console.error('일일 통계 로딩 실패:', err);
      setError(err.message || '데이터를 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // GPT 분석 요약 가져오기
  const fetchGptSummary = async (farmIdx, date) => {
    setGptLoading(true);
    setGptError(null);

    try {
      const data = await getDailyGptSummary(farmIdx, date);
      setGptSummary(data.summary);
    } catch (err) {
      console.error('GPT 분석 로딩 실패:', err);
      setGptError(err.message || 'GPT 분석을 불러오는데 실패했습니다.');
      setGptSummary('분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setGptLoading(false);
    }
  };

  // 농장이나 날짜가 변경될 때 자동으로 데이터 가져오기
  useEffect(() => {
    if (user?.selectedFarm?.farmIdx && selectedDate) {
      fetchDailyStats(selectedDate);
    }
  }, [user?.selectedFarm?.farmIdx, selectedDate]);

  return {
    stats,
    loading,
    error,
    gptSummary,
    gptLoading,
    gptError,
    refetch: fetchDailyStats
  };
};
