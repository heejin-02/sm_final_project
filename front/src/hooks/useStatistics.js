// src/hooks/useStatistics.js
import { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

/**
 * 통계 데이터를 가져오는 커스텀 훅
 * @param {string} period - 'daily', 'monthly', 'yearly'
 * @returns {Object} 통계 데이터
 */
export function useStatistics(period) {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user?.selectedFarm?.id || !period) return;

    const fetchStatistics = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // TODO: 실제 API 호출로 교체
        // const response = await axios.get(`/api/statistics/${period}`, {
        //   params: { farmId: user.selectedFarm.id },
        //   withCredentials: true
        // });
        // const result = response.data;
        
        // 임시 더미 데이터
        const dummyData = generateDummyData(period);
        
        // 실제 API 호출 시뮬레이션을 위한 딜레이
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setData(dummyData);
      } catch (err) {
        console.error('통계 데이터 로딩 실패:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [user?.selectedFarm?.id, period]);

  return { data, loading, error };
}

/**
 * 기간별 더미 데이터 생성
 * @param {string} period 
 * @returns {Object}
 */
function generateDummyData(period) {
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
        topRegion: 'B구역 (12마리)',
        detailList: [
          {
            id: 1,
            datetime: '2024-01-15 14:30',
            region: 'A구역',
            bugType: '진딧물',
            count: 5,
            accuracy: 95
          },
          {
            id: 2,
            datetime: '2024-01-15 16:45',
            region: 'B구역',
            bugType: '나방',
            count: 12,
            accuracy: 92
          },
          {
            id: 3,
            datetime: '2024-01-15 18:20',
            region: 'C구역',
            bugType: '거미',
            count: 6,
            accuracy: 88
          }
        ]
      };

    case 'monthly':
      return {
        ...baseData,
        totalDetections: 456,
        bugTypes: 8,
        topRegion: 'A구역 (156마리)',
        detailList: [
          {
            id: 1,
            datetime: '2024-01-01 ~ 2024-01-07',
            region: 'A구역',
            bugType: '진딧물',
            count: 89,
            accuracy: 94
          },
          {
            id: 2,
            datetime: '2024-01-08 ~ 2024-01-14',
            region: 'B구역',
            bugType: '나방',
            count: 67,
            accuracy: 91
          },
          {
            id: 3,
            datetime: '2024-01-15 ~ 2024-01-21',
            region: 'C구역',
            bugType: '거미',
            count: 45,
            accuracy: 89
          }
        ]
      };

    case 'yearly':
      return {
        ...baseData,
        totalDetections: 5234,
        bugTypes: 15,
        topRegion: 'B구역 (1,890마리)',
        detailList: [
          {
            id: 1,
            datetime: '2024년 1월',
            region: 'A구역',
            bugType: '진딧물',
            count: 1234,
            accuracy: 93
          },
          {
            id: 2,
            datetime: '2024년 2월',
            region: 'B구역',
            bugType: '나방',
            count: 987,
            accuracy: 90
          },
          {
            id: 3,
            datetime: '2024년 3월',
            region: 'C구역',
            bugType: '거미',
            count: 765,
            accuracy: 87
          }
        ]
      };

    default:
      return baseData;
  }
}
