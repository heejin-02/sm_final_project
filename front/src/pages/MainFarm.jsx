// src/pages/MainFarm.jsx
import React from 'react';
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
import NotiList from '../components/NotiList';

import { useRegionCounts } from "../hooks/useRegionCounts";
import FarmMap from "../components/FarmMap";
import Legend from "../components/Legend";
import Loader from "../components/Loader";

export default function MainFarm() {
  
  const { user } = useAuth();
  const farm = user?.selectedFarm;

  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  const { farmId } = useParams();
  const data = useRegionCounts(farmId);
  const navigate = useNavigate();

  if (!data) return <Loader message="구역별 탐지 내용을 불러오는 중입니다.." />;

  // 범례용 최소·최대값
  const counts = data.map(r => r.count);
  const min = Math.min(...counts), max = Math.max(...counts);
  
  return (
    <div className="main_farm p-4 flex gap-2 overflow-hidden">
      <NotiList/>
      <div className="right flex-1 w-full">
        <div className="farm_map w-full bg-emerald-700 p-2 rounded mb-4 relative">
          <Legend min={min} max={max} steps={5} />
          <div className="">
            <FarmMap
              data={data}
              rows={3}
              cols={3}
              cellSize={120}
              gap={8}
              onCellClick={id => navigate(`/regions/${id}`)}
            />
          </div>
        </div>
        <div className="flex gap-2 w-full">
          <div className="bordered-square-box">
            <span>탐지된 해충 수</span>
            <span>10마리</span>
          </div>          
          <div className="bordered-square-box">
            <span className="">탐지된 해충 종류</span>
            <span>5종</span>
          </div>       
          <div className="bordered-square-box">일간 통계</div>              
          <div className="bordered-square-box">월간 통계</div>              
          <div className="bordered-square-box">연간 통계</div>              
        </div>
      </div>
    </div>
  );
}
