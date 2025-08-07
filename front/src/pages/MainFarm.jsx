// src/pages/MainFarm.jsx
import { useNavigate, useParams } from "react-router-dom";
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
  const { farmIdx: urlFarmIdx } = useParams(); // URL에서 farmIdx 가져오기
  const { user } = useAuth();
  const farm = user?.selectedFarm;

  // URL 파라미터를 우선 사용, 없으면 selectedFarm 사용
  const farmIdx = urlFarmIdx ? parseInt(urlFarmIdx) : farm?.farmIdx;

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

  // 업데이트 상태 관리
  const [lastSummaryUpdate, setLastSummaryUpdate] = useState(null);
  const [lastMapUpdate, setLastMapUpdate] = useState(null);
  const [summaryStatus, setSummaryStatus] = useState('loading'); // 'loading', 'success', 'error', 'cached'
  const [mapStatus, setMapStatus] = useState('loading'); // 'loading', 'success', 'error', 'cached'

  // 시간 포맷팅 함수
  const getTimeAgo = (timestamp) => {
    if (!timestamp) return '업데이트 없음';

    const now = new Date();
    const diff = now - new Date(timestamp);
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));

    if (minutes < 1) return '방금 전';
    if (minutes < 60) return `${minutes}분 전`;
    if (hours < 24) return `${hours}시간 전`;
    return `${Math.floor(hours / 24)}일 전`;
  };

  // 상태별 메시지
  const getStatusMessage = (status) => {
    switch (status) {
      case 'loading': return '업데이트 중...';
      case 'success': return '최신 데이터';
      case 'error': return '연결 실패';
      case 'cached': return '캐시된 데이터';
      default: return '상태 확인 중...';
    }
  };



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
    console.log('🔥 useEffect 실행됨!', { farmIdx });

    const fetchAllData = async () => {

      if (!farmIdx) {
        return;
      }

      // 캐시 키 생성
      const summaryKey = `zone_summary_${farmIdx}`;
      const todayStatsKey = `today_stats_${farmIdx}`;
      const greenhouseKey = `greenhouse_data_${farmIdx}`;

      // 캐시된 데이터 확인 및 즉시 표시
      const cachedSummary = getData(summaryKey);
      const cachedTodayStats = getData(todayStatsKey);
      const cachedGreenhouse = getData(greenhouseKey);

      // 캐시된 데이터가 있으면 즉시 표시 (빠른 초기 로딩)
      if (cachedSummary) {
        setZoneSummary(cachedSummary);
        setSummaryStatus('cached');
        const cacheTimestamp = localStorage.getItem(`${summaryKey}_timestamp`);
        setLastSummaryUpdate(cacheTimestamp);
      } else {
        // 캐시가 없을 때만 초기 메시지 설정
        setZoneSummary("오늘 해충 탐지 정보를 분석 중입니다...");
        setSummaryStatus('loading');
      }

      // 캐시 여부와 관계없이 항상 최신 데이터 요청
      setSummaryLoading(true);

      // 캐시된 데이터가 있으면 즉시 표시
      if (cachedTodayStats) {
        setTodayStats(cachedTodayStats);
      }

      // 캐시 여부와 관계없이 항상 최신 데이터 요청
      setTodayStatsLoading(true);

      // 캐시된 데이터가 있으면 즉시 표시하고 로딩 완료 처리
      if (cachedGreenhouse) {
        setGreenhouseData(cachedGreenhouse);
        setMapStatus('cached');
        const cacheTimestamp = localStorage.getItem(`${greenhouseKey}_timestamp`);
        setLastMapUpdate(cacheTimestamp);
        setGreenhouseLoading(false); // 캐시가 있으면 로딩 완료 처리

        // 캐시가 5분 이내라면 API 호출 생략
        const cacheAge = cacheTimestamp ? Date.now() - new Date(cacheTimestamp).getTime() : Infinity;
        const CACHE_FRESH_DURATION = 5 * 60 * 1000; // 5분

        if (cacheAge < CACHE_FRESH_DURATION) {
          console.log('🏠 신선한 캐시 사용 - API 호출 생략 (캐시 나이:', Math.round(cacheAge / 1000), '초)');
          // 신선한 캐시가 있으면 greenhouse API 호출 생략
          var skipGreenhouseApi = true;
        }
      } else {
        // 캐시가 없을 때 기본 구역 데이터로 즉시 표시 (UX 개선)
        const defaultGreenhouses = Array.from({ length: 9 }, (_, i) => ({
          ghIdx: i + 1,
          ghName: `${i + 1}번 구역`,
          todayInsectCount: 0
        }));
        setGreenhouseData(defaultGreenhouses);
        setGreenhouseLoading(true); // 실제 데이터 로딩 중
      }

      // 모든 데이터가 캐시에 있어도 최신 데이터 요청 (실시간성 보장)
      console.log('📡 캐시와 관계없이 최신 데이터 요청 진행...');

      // 20초 후 강제 타임아웃 (너무 오래 기다리지 않도록)
      const timeoutId = setTimeout(() => {
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
      }, 20000);

      const today = new Date();
      const formattedDate = formatDateForAPI(today);

      try {
        // API 호출 배열 동적 구성
        const apiCalls = [
          retryApiCall(() => withTimeout(getDailyZoneSummary(farmIdx, formattedDate), 15000)), // 15초로 증가
          retryApiCall(() => withTimeout(getTodayStats(farmIdx), 5000)), // 5초로 증가
        ];

        // greenhouse API는 캐시가 신선하지 않을 때만 호출
        if (typeof skipGreenhouseApi === 'undefined' || !skipGreenhouseApi) {
          apiCalls.push(retryApiCall(() => withTimeout(getTodayGreenhouses(farmIdx), 5000))); // 5초로 증가
        }

        console.log('📡 API 호출 개수:', apiCalls.length, skipGreenhouseApi ? '(greenhouse API 생략)' : '');
        const startTime = Date.now();

        const results = await Promise.allSettled(apiCalls);

        // 결과 매핑 (greenhouse API 호출 여부에 따라)
        const [summaryResult, todayResult, greenhouseResult] = skipGreenhouseApi
          ? [results[0], results[1], { status: 'fulfilled', value: cachedGreenhouse }] // 캐시된 데이터 사용
          : results;

        const endTime = Date.now();
        console.log(`⏱️ API 호출 완료 (소요시간: ${endTime - startTime}ms)`);

        // 타임아웃 취소 (API 응답이 왔으므로)
        clearTimeout(timeoutId);

      if (summaryResult.status === 'fulfilled') {

        // API 응답 구조 확인 - 다양한 가능성 체크
        let newSummary = null;

        if (summaryResult.value && typeof summaryResult.value === 'object') {
          // 1. summary 필드가 있는 경우
          if (summaryResult.value.summary) {
            newSummary = summaryResult.value.summary;
          }
          // 2. data.summary 구조인 경우
          else if (summaryResult.value.data && summaryResult.value.data.summary) {
            newSummary = summaryResult.value.data.summary;
          }
          // 3. 응답 자체가 문자열인 경우
          else if (typeof summaryResult.value === 'string') {
            newSummary = summaryResult.value;
          }
          // 4. message 필드가 있는 경우
          else if (summaryResult.value.message) {
            newSummary = summaryResult.value.message;
          }
        }

        // 최종 검증
        if (!newSummary || newSummary.trim() === '') {
          newSummary = "오늘의 요약 정보가 없습니다.";
        }

        setZoneSummary(newSummary);
        setSummaryStatus('success');
        const now = new Date().toISOString();
        setLastSummaryUpdate(now);

        // 캐시에 저장 (타임스탬프 포함)
        setData(summaryKey, newSummary);
        localStorage.setItem(`${summaryKey}_timestamp`, now);
      } else {
        setSummaryStatus('error');
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
        setMapStatus('success');
        const now = new Date().toISOString();
        setLastMapUpdate(now);

        // 캐시에 저장 (일반 캐시 + 구역 전용 캐시 + 타임스탬프)
        setData(greenhouseKey, data);
        cacheGreenhouseData(farmIdx, data);
        localStorage.setItem(`${greenhouseKey}_timestamp`, now);

      } else {
        setMapStatus('error');
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
  if (!farm) return (
    <div className="select-farm section flex flex-col bg-[url('/images/home_bg.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center items-center justify-center">
          <div className="mt-4 space-y-4 text-white">
            <p className='text-2xl '>선택된 농장이 없습니다. <br/>다시 선택해주세요.</p>
          </div>    
      </div>
    </div>
  );

  // 온실 데이터 로딩 중일 때만 전체 로딩 표시 (다른 데이터는 부분적으로 표시)
  if (greenhouseLoading) {
    return (
      <div className="section flex flex-col items-center justify-center bg-gray-50">
        <div className="bg-white p-8 rounded-lg shadow-lg text-center">
          <div className="animate-spin w-12 h-12 border-4 border-gray-200 border-t-[#1A6900] rounded-full mx-auto mb-4"></div>
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
            <div className="flex flex-col w-full h-full gap-1">
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
              <div className="last-update">
                마지막 업데이트: {getTimeAgo(lastMapUpdate)} ({getStatusMessage(mapStatus)})
              </div>
            </div>
          </div>
          <div className="baekgu-msg-wrap h-[35%]">
            <div className="thumb">
              <img src="/images/talk_109.png" alt="" />
            </div>
            <div className="baekgu-msg flex flex-col w-full h-full gap-1">
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
              <div className="last-update">
                마지막 업데이트: {getTimeAgo(lastSummaryUpdate)} ({getStatusMessage(summaryStatus)})
              </div>
            </div>
          </div>

        </div>

        <div className="r-rt flex flex-col">
          <Weather/>
          {/* <div className="tit">아래와 같이 <br/>감지되었습니다.</div> */}
          <hr />
          <ul className="today_detecting flex-[1]">
            <li>
              <div className="today_tit">해충 수</div>
              <div className="today_desc">
                {todayStatsLoading ? (
                  <div className="flex items-center gap-1">
                    <span className="text-gray-400">--</span>
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
                    <span className="text-gray-400">--</span>
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
                    <span className="text-gray-400">--</span>
                  </div>
                ) : (
                  <b>{todayStats.zoneCount}</b>
                )} {!todayStatsLoading && '곳'}
              </div>
            </li>
          </ul>
          <hr />
          <div className='day-check-buttons flex-[1]'>
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
