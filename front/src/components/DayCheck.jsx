// src/components/DayCheck.jsx
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export default function DayCheck() {
  const navigate = useNavigate();
  const location = useLocation();

  // 현재 경로에서 period 추출
  const getCurrentPeriod = () => {
    if (location.pathname.includes('/report/daily')) return 'daily';
    if (location.pathname.includes('/report/monthly')) return 'monthly';
    if (location.pathname.includes('/report/yearly')) return 'yearly';
    return null;
  };

  const currentPeriod = getCurrentPeriod();

  // TODO: 실제 데이터로 교체
  const todayStats = {
    total: 5,
    regions: 2,
    types: 2
  };

  return (
    <div className="day-check scrl-custom">
      {/* 오늘의 해충 타이틀 */}
      <div className="day-check-header">
        <h3 className="tit">오늘 찾은 해충</h3>
      </div>

      {/* 통계 박스들 */}
      <div className="day-check-stats">
        <div className="stat-box">
          <div className="stat-label">찾은 해충</div>
          <div className="stat-unit">
              <span className="stat-value">{todayStats.total}</span> 마리
          </div>
        </div>
        
        <div className="stat-box">
          <div className="stat-label">탐지 종류</div>
          <div className="stat-unit">
              <span className="stat-value">{todayStats.regions}</span> 종
          </div>
        </div>
        
        <div className="stat-box">
          <div className="stat-label">발생 구역</div>
          <div className="stat-unit">
              <span className="stat-value">{todayStats.types}</span> 곳
          </div>
        </div>
      </div>

      {/* 통계 버튼들 */}
      <div className="day-check-buttons">
        <button
          className={`btn stat-button hvborder ${currentPeriod === 'daily' ? 'active' : ''}`}
          onClick={() => navigate('/report/daily')}
        >
          일간 통계
        </button>
        <button
          className={`btn stat-button hvborder ${currentPeriod === 'monthly' ? 'active' : ''}`}
          onClick={() => navigate('/report/monthly')}
        >
          월간 통계
        </button>
        <button
          className={`btn stat-button hvborder ${currentPeriod === 'yearly' ? 'active' : ''}`}
          onClick={() => navigate('/report/yearly')}
        >
          연간 통계
        </button>
      </div>
    </div>
  );
}
