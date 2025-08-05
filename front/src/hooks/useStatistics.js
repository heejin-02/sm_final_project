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
    if (!user?.selectedFarm?.farmIdx || !period) return;

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

        // console.log(`📊 [Statistics] API 호출: ${period}, 날짜: ${formatDate(selectedDate)}, farmIdx: ${user.selectedFarm.farmIdx}`);

        // TODO: 실제 API 호출로 교체
        // const response = await axios.get(`/api/statistics/${period}`, {
        //   params: {
        //     farmIdx: user.selectedFarm.farmIdx,
        //     date: formatDate(selectedDate)
        //   },
        //   withCredentials: true
        // });
        // const result = response.data;

        // 임시 더미 데이터 (선택된 날짜 반영)
        const dummyData = generateDummyData(period, selectedDate);

        // 실제 API 호출 시뮬레이션을 위한 딜레이
        await new Promise(resolve => setTimeout(resolve, 500));

        setData(dummyData);
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

/**
 * 기간별 더미 데이터 생성
 * @param {string} period
 * @param {Date} selectedDate
 * @returns {Object}
 */
function generateDummyData(period, selectedDate = new Date()) {
  const baseData = {
    totalDetections: 0,
    bugTypes: 0,
    topRegion: '',
    detailList: []
  };

  switch (period) {
    case 'daily':
      return {
        ...baseData,
        totalDetections: 23,
        bugTypes: 4,
        topRegion: '1번 온실 (12건)',
        detailList: [
          {
            anlsIdx: 999999991,
            datetime: '2025-01-31 14:30',
            region: '1번 온실',
            bugType: '알락수염노린재',
            count: 5,
            accuracy: 85
          },
          {
            anlsIdx: 999999992,
            datetime: '2025-01-31 16:45',
            region: '2번 온실',
            bugType: '꽃노랑총채벌레',
            count: 8,
            accuracy: 92
          },
          {
            anlsIdx: 999999993,
            datetime: '2025-01-31 18:20',
            region: '3번 온실',
            bugType: '담배가루이',
            count: 3,
            accuracy: 78
          },
          {
            anlsIdx: 999999994,
            datetime: '2025-01-31 20:15',
            region: '1번 온실',
            bugType: '비단노린재',
            count: 7,
            accuracy: 88
          }
        ]
      };

    case 'monthly':
      return {
        ...baseData,
        totalDetections: 456,
        bugTypes: 4,
        topRegion: '1번 온실 (156건)',
        groupedData: {
          'week1': {
            title: '2025년 1월 1주차 (1일~7일)',
            count: 89,
            items: [
              { anlsIdx: 999999991, datetime: '2025-01-02 14:30', region: '1번 온실', bugType: '알락수염노린재', accuracy: 85 },
              { anlsIdx: 999999992, datetime: '2025-01-03 16:45', region: '2번 온실', bugType: '꽃노랑총채벌레', accuracy: 92 },
              { anlsIdx: 999999993, datetime: '2025-01-05 18:20', region: '3번 온실', bugType: '담배가루이', accuracy: 78 }
            ]
          },
          'week2': {
            title: '2025년 1월 2주차 (8일~14일)',
            count: 67,
            items: [
              { anlsIdx: 999999994, datetime: '2025-01-09 10:15', region: '1번 온실', bugType: '비단노린재', accuracy: 88 },
              { anlsIdx: 999999995, datetime: '2025-01-11 15:30', region: '2번 온실', bugType: '알락수염노린재', accuracy: 86 }
            ]
          },
          'week3': {
            title: '2025년 1월 3주차 (15일~21일)',
            count: 45,
            items: [
              { anlsIdx: 999999996, datetime: '2025-01-16 12:45', region: '3번 온실', bugType: '꽃노랑총채벌레', accuracy: 91 },
              { anlsIdx: 999999997, datetime: '2025-01-19 17:20', region: '1번 온실', bugType: '담배가루이', accuracy: 79 }
            ]
          },
          'week4': {
            title: '2025년 1월 4주차 (22일~31일)',
            count: 255,
            items: [
              { anlsIdx: 999999998, datetime: '2025-01-24 09:30', region: '2번 온실', bugType: '비단노린재', accuracy: 87 },
              { anlsIdx: 999999999, datetime: '2025-01-28 14:15', region: '1번 온실', bugType: '알락수염노린재', accuracy: 84 }
            ]
          }
        }
      };

    case 'yearly':
      return {
        ...baseData,
        totalDetections: 5234,
        bugTypes: 4,
        topRegion: '1번 온실 (1,890건)',
        groupedData: {
          'month1': {
            title: '2024년 1월',
            count: 1234,
            items: [
              { anlsIdx: 999999991, datetime: '2024-01-05 14:30', region: '1번 온실', bugType: '알락수염노린재', accuracy: 85 },
              { anlsIdx: 999999992, datetime: '2024-01-12 16:45', region: '2번 온실', bugType: '꽃노랑총채벌레', accuracy: 92 },
              { anlsIdx: 999999993, datetime: '2024-01-20 18:20', region: '3번 온실', bugType: '담배가루이', accuracy: 78 }
            ]
          },
          'month2': {
            title: '2024년 2월',
            count: 987,
            items: [
              { anlsIdx: 999999994, datetime: '2024-02-03 10:15', region: '1번 온실', bugType: '비단노린재', accuracy: 88 },
              { anlsIdx: 999999995, datetime: '2024-02-15 15:30', region: '2번 온실', bugType: '알락수염노린재', accuracy: 86 }
            ]
          },
          'month3': {
            title: '2024년 3월',
            count: 765,
            items: [
              { anlsIdx: 999999996, datetime: '2024-03-08 12:45', region: '3번 온실', bugType: '꽃노랑총채벌레', accuracy: 91 },
              { anlsIdx: 999999997, datetime: '2024-03-22 17:20', region: '1번 온실', bugType: '담배가루이', accuracy: 79 }
            ]
          },
          'month4': {
            title: '2024년 4월',
            count: 2248,
            items: [
              { anlsIdx: 999999998, datetime: '2024-04-10 09:30', region: '2번 온실', bugType: '비단노린재', accuracy: 87 },
              { anlsIdx: 999999999, datetime: '2024-04-25 14:15', region: '1번 온실', bugType: '알락수염노린재', accuracy: 84 }
            ]
          }
        }
      };

    default:
      return baseData;
  }
}
