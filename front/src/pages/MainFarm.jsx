// src/pages/MainFarm.jsx
import React from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function MainFarm() {
  const { user } = useAuth();

  const farm = user?.selectedFarm;

  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">{farm.name} 관리 페이지</h2>
      <p>농장 ID: {farm.id}</p>
      {/* 추가적인 농장 정보 출력 또는 관리 기능 */}
    </div>
  );
}
