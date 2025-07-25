// src/components/DayCheck.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function DayCheck() {
  const navigate = useNavigate();

  // TODO: 실제 데이터로 교체
  const todayStats = {
    total: 5,
    regions: 2,
    types: 2
  };

  const advice = "오늘은 어제 보다 벌레가 없네요! 다만 B 구역에서만 4 마리의 벌레가 발견 되어 확인이 필요 합니다. 다음 주에 비가 내릴 예정 입니다. 습도 관리에 신경써 주세요. 백구가 계속 지켜 볼게요 왕!";

  return (
    <div className="day-check">
      {/* 오늘의 해충 타이틀 */}
      <div className="day-check-header">
        <h3 className="tit">오늘 찾은 해충이다멍!</h3>
      </div>

      {/* 통계 박스들 */}
      <div className="day-check-stats">
        <div className="stat-box">
          <div className="stat-label">해충 발견</div>
          <div className="stat-value">{todayStats.total}</div>
          <div className="stat-unit">마리</div>
        </div>
        
        <div className="stat-box">
          <div className="stat-label">발견 구역</div>
          <div className="stat-value">{todayStats.regions}</div>
          <div className="stat-unit">곳</div>
        </div>
        
        <div className="stat-box">
          <div className="stat-label">발견 종류</div>
          <div className="stat-value">{todayStats.types}</div>
          <div className="stat-unit">종</div>
        </div>
      </div>

      {/* 통계 버튼들 */}
      <div className="day-check-buttons">
        <button className="stat-button active" onClick={() => navigate('/report/daily')}>일간 통계</button>
        <button className="stat-button" onClick={() => navigate('/report/monthly')}>월간 통계</button>
        <button className="stat-button" onClick={() => navigate('/report/yearly')}>연간 통계</button>
      </div>
    </div>
  );
}
