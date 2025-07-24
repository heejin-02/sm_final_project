// src/pages/MainFarm.jsx
import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import NotiList from '../components/NotiList';
import FarmMap from '../components/FarmMap';
import { getUserFarms } from '../api/auth';

export default function MainFarm() {
  const { user } = useAuth();

  const farm = user?.selectedFarm;

  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  return (
    <div className="main_farm p-4 flex gap-2 overflow-hidden">
      <NotiList/>
      <div className="right">
        <FarmMap/>
        <div className="flex gap-2">
          <div>
            <span>탐지된 해충 수</span>
            <span>10마리</span>
          </div>          
          <div>
            <span>탐지된 해충 종류</span>
            <span>5종</span>
          </div>          
        </div>
      </div>
    </div>
  );
}
