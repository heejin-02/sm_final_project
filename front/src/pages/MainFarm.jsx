// src/pages/MainFarm.jsx
import { useNavigate } from "react-router-dom";
import { useState, useEffect, useMemo, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useDataCache } from '../contexts/DataCacheContext';
import { getDailyZoneSummary, formatDateForAPI, getTodayStats, getTodayGreenhouses } from '../api/report';
import { retryApiCall, withTimeout } from '../utils/apiRetry';
import NotiList from '../components/NotiList';
import Weather from '../components/Weather';
import BaseFarmMap from "../components/BaseFarmMap";
import Legend from "../components/Legend";
  
export default function MainFarm() {

  const { user } = useAuth();
  const farm = user?.selectedFarm;
  const farmIdx = farm?.farmIdx;
  const { getData, setData, setGreenhouseData: cacheGreenhouseData } = useDataCache();

  // 백구 메시지 상태 - localStorage에서 초기값 복원
  const [zoneSummary, setZoneSummary] = useState(() => {
    try {
      const cached = localStorage.getItem(`zone_summary_${farmIdx}`);
      return cached || "오늘 해충 탐지 정보를 분석 중입니다...";
    } catch {
      return "오늘 해충 탐지 정보를 분석 중입니다...";
    }
  });
  const [summaryLoading, setSummaryLoading] = useState(true);

  // 오늘 통계 상태 - localStorage에서 초기값 복원
  const [todayStats, setTodayStats] = useState(() => {
    try {
      const cached = localStorage.getItem(`today_stats_${farmIdx}`);
      return cached ? JSON.parse(cached) : { todayCount: 0, insectTypeCount: 0, zoneCount: 0 };
    } catch {
      return { todayCount: 0, insectTypeCount: 0, zoneCount: 0 };
    }
  });
  const [todayStatsLoading, setTodayStatsLoading] = useState(true);

  // 온실별 데이터 상태 - localStorage에서 초기값 복원
  const [greenhouseData, setGreenhouseData] = useState(() => {
    try {
      const cached = localStorage.getItem(`greenhouse_data_${farmIdx}`);
      return cached ? JSON.parse(cached) : [];
    } catch {
      return [];
    }
  });
  const [greenhouseLoading, setGreenhouseLoading] = useState(true);

  // Hook들을 조건부 return보다 먼저 호출
  const navigate = useNavigate();

  // 네비게이션 함수들 - useCallback으로 최적화
  const handleDailyReport = useCallback(() => {
    navigate('/report/daily');
  }, [navigate]);

  const handleMonthlyReport = useCallback(() => {
    navigate('/report/monthly');
  }, [navigate]);

  const handleYearlyReport = useCallback(() => {
    navigate('/report/yearly');
  }, [navigate]);

  // 범례용 최소·최대값 (온실 데이터 기반) - 메모이제이션으로 성능 최적화
  const { min, max } = useMemo(() => {
    const counts = greenhouseData.length > 0 ? greenhouseData.map(r => r.todayInsectCount || 0) : [0];
    return {
      min: Math.min(...counts),
      max: Math.max(...counts)
    };
  }, [greenhouseData]);

  // 모든 데이터 병렬로 가져오기 (빠른 로딩 + 캐시 활용)
  useEffect(() => {
    const fetchAllData = async () => {
      if (!farmIdx) return;

      // 캐시 키 생성
      const summaryKey = `zone_summary_${farmIdx}`;
      const todayStatsKey = `today_stats_${farmIdx}`;
      const greenhouseKey = `greenhouse_data_${farmIdx}`;

      console.log('🗂️ 캐시 키들:', { summaryKey, todayStatsKey, greenhouseKey });

      // 캐시된 데이터 확인 및 즉시 표시
      const cachedSummary = getData(summaryKey);
      const cachedTodayStats = getData(todayStatsKey);
      const cachedGreenhouse = getData(greenhouseKey);

      console.log('💾 캐시된 데이터:', {
        cachedSummary,
        cachedTodayStats: !!cachedTodayStats,
        cachedGreenhouse: !!cachedGreenhouse
      });

      if (cachedSummary) {
        setZoneSummary(cachedSummary);
        setSummaryLoading(false);
      } else {
        setSummaryLoading(true);
      }

      if (cachedTodayStats) {
        setTodayStats(cachedTodayStats);
        setTodayStatsLoading(false);
      } else {
        setTodayStatsLoading(true);
      }

      if (cachedGreenhouse) {
        setGreenhouseData(cachedGreenhouse);
        setGreenhouseLoading(false);
      } else {
        setGreenhouseLoading(true);
      }

      // 모든 데이터가 캐시에 있으면 API 호출 생략
      if (cachedSummary && cachedTodayStats && cachedGreenhouse) {
        // console.log('모든 데이터 캐시에서 로드 완료');
        return;
      }

      // 8초 후 강제 타임아웃 (너무 오래 기다리지 않도록)
      const timeoutId = setTimeout(() => {
        // console.log('API 타임아웃 - fallback 데이터 사용');
        setSummaryLoading(false);
        setTodayStatsLoading(false);
        setGreenhouseLoading(false);

        setZoneSummary("서버 연결이 지연되고 있습니다. 잠시 후 다시 시도해주세요.");
        setTodayStats({ todayCount: 0, insectTypeCount: 0, zoneCount: 0 });
        setGreenhouseData([
          { ghIdx: 1, ghName: "1번 구역", todayInsectCount: 0 },
          { ghIdx: 2, ghName: "2번 구역", todayInsectCount: 0 },
          { ghIdx: 3, ghName: "3번 구역", todayInsectCount: 0 },
          { ghIdx: 4, ghName: "4번 구역", todayInsectCount: 0 },
          { ghIdx: 5, ghName: "5번 구역", todayInsectCount: 0 },
          { ghIdx: 6, ghName: "6번 구역", todayInsectCount: 0 },
          { ghIdx: 7, ghName: "7번 구역", todayInsectCount: 0 },
          { ghIdx: 8, ghName: "8번 구역", todayInsectCount: 0 },
          { ghIdx: 9, ghName: "9번 구역", todayInsectCount: 0 }
        ]);
      }, 8000);

      const today = new Date();
      const formattedDate = formatDateForAPI(today);

      console.log('🔍 MainFarm API 호출 준비:', {
        farmIdx,
        today,
        formattedDate,
        user: user?.selectedFarm
      });

      try {
        // 모든 API를 병렬로 호출 (재시도 + 타임아웃 적용)
        console.log('📡 API 호출 시작...');
        const [summaryResult, todayResult, greenhouseResult] = await Promise.allSettled([
          retryApiCall(() => withTimeout(getDailyZoneSummary(farmIdx, formattedDate), 4000)),
          retryApiCall(() => withTimeout(getTodayStats(farmIdx), 4000)),
          retryApiCall(() => withTimeout(getTodayGreenhouses(farmIdx), 4000))
        ]);

        console.log('📥 API 응답 완료:', {
          summaryResult: summaryResult.status,
          todayResult: todayResult.status,
          greenhouseResult: greenhouseResult.status
        });

        // 타임아웃 취소 (API 응답이 왔으므로)
        clearTimeout(timeoutId);

      // 1. 백구 메시지 결과 처리
      console.log('🐕 백구 메시지 처리 시작:', summaryResult);

      if (summaryResult.status === 'fulfilled') {
        console.log('✅ summaryResult.value:', summaryResult.value);
        console.log('✅ summaryResult.value.summary:', summaryResult.value.summary);
        console.log('✅ typeof summary:', typeof summaryResult.value.summary);

        const newSummary = summaryResult.value.summary || "오늘의 요약 정보가 없습니다.";
        console.log('✅ 최종 newSummary:', newSummary);

        setZoneSummary(newSummary);
        // 캐시에 저장
        setData(summaryKey, newSummary);
      } else {
        console.error('🚨 구역별 요약 데이터 로딩 실패:', summaryResult.reason);
        console.error('🚨 summaryResult 전체:', summaryResult);
        if (!cachedSummary) {
          setZoneSummary("요약 정보를 불러오는데 실패했습니다. 잠시 후 다시 시도해주세요.");
        }
      }
      setSummaryLoading(false);

      // 2. 오늘 통계 결과 처리
      if (todayResult.status === 'fulfilled') {
        const newStats = {
          todayCount: todayResult.value.todayCount || 0,
          insectTypeCount: todayResult.value.insectTypeCount || 0,
          zoneCount: todayResult.value.zoneCount || 0
        };
        setTodayStats(newStats);
        // 캐시에 저장
        setData(todayStatsKey, newStats);
      } else {
        console.error('오늘 통계 데이터 로딩 실패:', todayResult.reason);
        if (!cachedTodayStats) {
          // API 실패 시 더미 데이터 사용
          const fallbackStats = {
            todayCount: 0,
            insectTypeCount: 0,
            zoneCount: 0
          };
          setTodayStats(fallbackStats);
        }
      }
      setTodayStatsLoading(false);

      // 3. 온실별 데이터 결과 처리
      if (greenhouseResult.status === 'fulfilled') {
        const data = greenhouseResult.value || [];
        setGreenhouseData(data);
        // 캐시에 저장 (일반 캐시 + 구역 전용 캐시)
        setData(greenhouseKey, data);
        cacheGreenhouseData(farmIdx, data);
      } else {
        console.error('온실별 데이터 로딩 실패:', greenhouseResult.reason);
        if (!cachedGreenhouse) {
          // API 실패 시 더미 데이터 사용
          const fallbackGreenhouses = [
            { ghIdx: 1, ghName: "1번 구역", todayInsectCount: 0 },
            { ghIdx: 2, ghName: "2번 구역", todayInsectCount: 0 },
            { ghIdx: 3, ghName: "3번 구역", todayInsectCount: 0 },
            { ghIdx: 4, ghName: "4번 구역", todayInsectCount: 0 },
            { ghIdx: 5, ghName: "5번 구역", todayInsectCount: 0 },
            { ghIdx: 6, ghName: "6번 구역", todayInsectCount: 0 },
            { ghIdx: 7, ghName: "7번 구역", todayInsectCount: 0 },
            { ghIdx: 8, ghName: "8번 구역", todayInsectCount: 0 },
            { ghIdx: 9, ghName: "9번 구역", todayInsectCount: 0 }
          ];
          setGreenhouseData(fallbackGreenhouses);
        }
      }
      setGreenhouseLoading(false);

      } catch (error) {
        // 전체 API 호출 실패 시
        console.error('전체 API 호출 실패:', error);
        clearTimeout(timeoutId);

        setSummaryLoading(false);
        setTodayStatsLoading(false);
        setGreenhouseLoading(false);

        setZoneSummary("서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.");
        setTodayStats({ todayCount: 0, insectTypeCount: 0, zoneCount: 0 });
        setGreenhouseData([]);
      }
    };

    fetchAllData();
  }, [farmIdx]);

  // 조건부 return은 모든 Hook 호출 후에
  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  // 온실 데이터 로딩 중일 때만 전체 로딩 표시 (다른 데이터는 부분적으로 표시)
  if (greenhouseLoading) {
    return (
      <div className="section flex flex-col items-center justify-center bg-gray-50">
        <div className="bg-white p-8 rounded-lg shadow-lg text-center">
          <div className="animate-spin w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full mx-auto mb-4"></div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">농장 데이터 로딩 중</h3>
          <p className="text-gray-500">잠시만 기다려주세요...</p>
          <div className="mt-4 text-sm text-gray-400">
            {summaryLoading && todayStatsLoading ? '초기 로딩 중...' : '거의 완료되었습니다!'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="main-farm p-4 flex gap-4 overflow-hidden">
      <NotiList/>

      <div className="right flex-1 w-full flex gap-4">
      
        <div className="r-lt">
          <div className="farm_map h-[65%]">
            <Legend min={min} max={max} />
            <BaseFarmMap
              mode="overview"
              data={[]}
              greenhouseData={greenhouseData}
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
                    <div className="flex items-center gap-2 text-gray-500">
                      <div className="animate-spin w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full"></div>
                      <span>백구가 농장 데이터를 분석하고 있습니다...</span>
                    </div>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{zoneSummary}</div>
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
              <div className="today_desc">
                {todayStatsLoading ? (
                  <div className="flex items-center gap-1">
                    <div className="animate-pulse w-6 h-6 bg-gray-200 rounded"></div>
                    <span className="text-gray-400">로딩중</span>
                  </div>
                ) : (
                  <b>{todayStats.todayCount}</b>
                )} {!todayStatsLoading && '마리'}
              </div>
            </li>
            <li>
              <div className="today_tit">해충 종류</div>
              <div className="today_desc">
                {todayStatsLoading ? (
                  <div className="flex items-center gap-1">
                    <div className="animate-pulse w-6 h-6 bg-gray-200 rounded"></div>
                    <span className="text-gray-400">로딩중</span>
                  </div>
                ) : (
                  <b>{todayStats.insectTypeCount}</b>
                )} {!todayStatsLoading && '종'}
              </div>
            </li>
            <li>
              <div className="today_tit">발견된 구역</div>
              <div className="today_desc">
                {todayStatsLoading ? (
                  <div className="flex items-center gap-1">
                    <div className="animate-pulse w-6 h-6 bg-gray-200 rounded"></div>
                    <span className="text-gray-400">로딩중</span>
                  </div>
                ) : (
                  <b>{todayStats.zoneCount}</b>
                )} {!todayStatsLoading && '곳'}
              </div>
            </li>
          </ul>
          <hr />
          <div className='day-check-buttons'>
            <div
              className="btn stat-button"
              onClick={handleDailyReport}
            >
              일간 통계
            </div>
            <div
              className="btn stat-button"
              onClick={handleMonthlyReport}
            >
              월간 통계
            </div>
            <div
              className="btn stat-button"
              onClick={handleYearlyReport}
            >
              연간 통계
            </div>
          </div>     
        </div>

      </div>
    
    </div>
  );
}
