// src/pages/MainFarm.jsx
import { useNavigate } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
import NotiList from '../components/NotiList';

import { useRegionCounts } from "../hooks/useRegionCounts";
import BaseFarmMap from "../components/BaseFarmMap";
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
    <div className="main-farm p-4 flex gap-4 overflow-hidden">
      <NotiList/>
      <div className="right flex-1 w-full flex gap-4">
        <div className="r-lt">
          <div className="farm_map h-[65%]">
            <Legend min={min} max={max} />            
            <BaseFarmMap
              mode="overview"
              data={data}
              rows={3}
              cols={3}
              gap={0}
              showHeatmap={true}
              interactive={false}
              // onCellClick={id => navigate(`/regions/${id}`)}
            />
          </div>
          <div className="baekgu-msg-wrap h-[35%]">
            <div className="thumb">
              <img src="/images/talk_109.png" alt="" />
            </div>
            <div className="baekgu-msg">
              <div className="scrl-custom overflow-y-auto h-full">
                오늘은 어제 보다 벌레가 없네요! 다만 B 구역 에서만 4 마리의 벌레가 발견 되어 확인이 필요 합니다. 오후에 비가 내릴 예정 입니다. 습도 관리에 신경 써 주세요.
              </div>
            </div>
          </div>          

        </div>
        <div className="r-rt">
          <div className="tit">아래와 같이 <br/>감지되었습니다.</div>
          <hr />

          <hr />
        </div>
      </div>
    </div>
  );
}
