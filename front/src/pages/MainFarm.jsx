// src/pages/MainFarm.jsx
import { useNavigate } from "react-router-dom";
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

  const farmIdx = farm.farmIdx; // AuthContext에서 farmIdx 가져오기
  const data = useRegionCounts(farmIdx);
  const navigate = useNavigate();

  if (!data) return <Loader message="구역별 탐지 내용을 불러오는 중입니다.." />;

  // 범례용 최소·최대값
  const counts = data.map(r => r.count);
  const min = Math.min(...counts), max = Math.max(...counts);
  
  return (
    <div className="main_farm p-4 flex gap-2 overflow-hidden">
      <NotiList/>
      <div className="right flex-1 w-full">
        <div className="farm_map">
          <Legend min={min} max={max} />
          <div className="h-full">
            <FarmMap
              data={data}
              rows={3}
              cols={3}
              gap={8}
              // onCellClick={id => navigate(`/regions/${id}`)}
            />
          </div>
        </div>
        <div className="stats-container">
          <div className="stats-column">
            <div className="stat-item">
              <span className="stat-label">찾은 해충</span>
              <span className="stat-content">
                <span className="stat-number">10</span><span className="stat-unit"> 마리</span>
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">탐지 종류</span>
              <span className="stat-content">
                <span className="stat-number">5</span><span className="stat-unit"> 종</span>
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">발생 구역</span>
              <span className="stat-content">
                <span className="stat-number">5</span><span className="stat-unit"> 곳</span>
              </span>
            </div>
          </div>

          <div className="bordered-box justify-start overflow-y-auto scrl-custom">
            오늘은 어제 보다 벌레가 없네요! 다만 B 구역 에서만 4 마리의 벌레가 발견 되어 확인이 필요 합니다. 오후에 비가 내릴 예정 입니다. 습도 관리에 신경 써 주세요.
          </div>

          <div className='flex flex-col gap-1 btn-wrap'>
            <div
              className="bordered-box hvborder cursor-pointer bg-[#0066c5]"
              onClick={() => navigate('/report/daily')}
            >
              일간 통계
            </div>
            <div
              className="bordered-box hvborder cursor-pointer"
              onClick={() => navigate('/report/monthly')}
            >
              월간 통계
            </div>
            <div
              className="bordered-box hvborder cursor-pointer bg-[#00488a]"
              onClick={() => navigate('/report/yearly')}
            >
              연간 통계
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
