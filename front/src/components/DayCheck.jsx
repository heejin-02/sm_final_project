// src/components/DayCheck.jsx
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { getTodayStats } from '../api/report';

export default function DayCheck() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // 오늘 통계 상태
  const [todayStats, setTodayStats] = useState({
    todayCount: 0,
    insectTypeCount: 0,
    zoneCount: 0
  });
  const [loading, setLoading] = useState(true);

  // 현재 경로에서 period 추출
  const getCurrentPeriod = () => {
    if (location.pathname.includes('/report/daily')) return 'daily';
    if (location.pathname.includes('/report/monthly')) return 'monthly';
    if (location.pathname.includes('/report/yearly')) return 'yearly';
    return null;
  };

  const currentPeriod = getCurrentPeriod();

  // 오늘 통계 데이터 가져오기
  useEffect(() => {
    const fetchTodayStats = async () => {
      if (!user?.selectedFarm?.farmIdx) return;

      try {
        setLoading(true);
        const response = await getTodayStats(user.selectedFarm.farmIdx);
        setTodayStats({
          todayCount: response.todayCount || 0,
          insectTypeCount: response.insectTypeCount || 0,
          zoneCount: response.zoneCount || 0
        });
      } catch (error) {
        console.error('오늘 통계 데이터 로딩 실패:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTodayStats();
  }, [user?.selectedFarm?.farmIdx]);

  return (
    <div className="day-check">
      {/* 오늘의 해충 타이틀 */}
      <div className="day-check-header">
        <h3 className="tit">오늘 찾은 해충</h3>
      </div>

      {/* 통계 박스들 */}
      <div className="day-check-stats">
        <div className="stat-box">
          <div className="stat-label">해충 수</div>
          <div className="stat-unit">
              <span className="stat-value">
                {loading ? '--' : todayStats.todayCount}
              </span> 마리
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">해충 종류</div>
          <div className="stat-unit">
              <span className="stat-value">
                {loading ? '--' : todayStats.insectTypeCount}
              </span> 종
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">발견된 구역</div>
          <div className="stat-unit">
              <span className="stat-value">
                {loading ? '--' : todayStats.zoneCount}
              </span> 곳
          </div>
        </div>
      </div>

      {/* 통계 버튼들 */}
      <div className="day-check-buttons">
        <button
          className={`btn stat-button ${currentPeriod === 'daily' ? 'active' : ''}`}
          onClick={() => navigate('/report/daily')}
        >
          일간 통계
        </button>
        <button
          className={`btn stat-button ${currentPeriod === 'monthly' ? 'active' : ''}`}
          onClick={() => navigate('/report/monthly')}
        >
          월간 통계
        </button>
        <button
          className={`btn stat-button ${currentPeriod === 'yearly' ? 'active' : ''}`}
          onClick={() => navigate('/report/yearly')}
        >
          연간 통계
        </button>
      </div>
    </div>
  );
}
