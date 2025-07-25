// src/components/LeftPanel.jsx
import React from 'react';
import NotiList from './NotiList';
import DayCheck from './DayCheck';

export default function LeftPanel() {
  return (
    <div className="left-panel">
      {/* 확인하지 않은 알림 섹션 */}
      <div className="left-panel-section h-[60%] overflow-y-hidden">
        <NotiList />
      </div>
      
      {/* 오늘의 해충 섹션 */}
      <div className="left-panel-section h-fit scrl-custom">
        <DayCheck />
      </div>
    </div>
  );
}
