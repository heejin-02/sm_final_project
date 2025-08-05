// src/pages/MainFarm.jsx
import { useNavigate } from "react-router-dom";
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { getDailyZoneSummary, formatDateForAPI } from '../api/report';
import NotiList from '../components/NotiList';
import Weather from '../components/Weather';
import { LuLogOut } from "react-icons/lu";

import { useRegionCounts } from "../hooks/useRegionCounts";
import BaseFarmMap from "../components/BaseFarmMap";
import Legend from "../components/Legend";
import Loader from "../components/Loader";

export default function MainFarm() {

  const { user } = useAuth();
  const farm = user?.selectedFarm;

  // 백구 메시지 상태
  const [zoneSummary, setZoneSummary] = useState("데이터를 불러오는 중입니다...");
  const [summaryLoading, setSummaryLoading] = useState(true);

  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  const farmIdx = farm.farmIdx; // AuthContext에서 farmIdx 가져오기
  const data = useRegionCounts(farmIdx);
  const navigate = useNavigate();

  // 백구 메시지 데이터 가져오기
  useEffect(() => {
    const fetchZoneSummary = async () => {
      try {
        setSummaryLoading(true);
        // 어제 날짜로 요약 가져오기 (오늘은 아직 끝나지 않았으므로)
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        const formattedDate = formatDateForAPI(yesterday);

        const response = await getDailyZoneSummary(farmIdx, formattedDate);
        setZoneSummary(response.summary_text || "오늘의 요약 정보가 없습니다.");
      } catch (error) {
        console.error('구역별 요약 데이터 로딩 실패:', error);
        setZoneSummary("요약 정보를 불러오는데 실패했습니다. 잠시 후 다시 시도해주세요.");
      } finally {
        setSummaryLoading(false);
      }
    };

    if (farmIdx) {
      fetchZoneSummary();
    }
  }, [farmIdx]);

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
                {summaryLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-gray-500">백구가 분석 중입니다...</div>
                  </div>
                ) : (
                  zoneSummary
                )}
              </div>
            </div>
          </div>

        </div>

        <div className="r-rt">
          <Weather/>
          {/* <div className="tit">아래와 같이 <br/>감지되었습니다.</div> */}
          <hr />
          <ul className="today_detecting">
            <li>
              <div className="today_tit">해충 수</div>
              <div className="today_desc"><b>177</b> 마리</div>
            </li>
            <li>
              <div className="today_tit">해충 종류</div>
              <div className="today_desc"><b>3</b> 종</div>
            </li>
            <li>
              <div className="today_tit">발견된 구역</div>
              <div className="today_desc"><b>11</b> 곳</div>
            </li>                        
          </ul>
          <hr />
          <div className='day-check-buttons'>
            <div
              className="btn stat-button"
              onClick={() => navigate('/report/daily')}
            >
              일간 통계
            </div>
            <div
              className="btn stat-button"
              onClick={() => navigate('/report/monthly')}
            >
              월간 통계
            </div>
            <div
              className="btn stat-button"
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
