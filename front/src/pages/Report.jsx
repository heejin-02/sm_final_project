// src/pages/Report.jsx
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useStatistics } from '../hooks/useStatistics';
import { useDailyStats } from '../hooks/useDailyStats';
import Loader from '../components/Loader';
import LeftPanel from '../components/LeftPanel';
import DateNavigation from '../components/DateNavigation';
import GroupedDetailList from '../components/GroupedDetailList';
import StatisticsChart from '../components/StatisticsChart';

export default function Report() {
  const { period } = useParams(); // 'daily', 'monthly', 'yearly'
  const { user } = useAuth();
  // daily 모드일 때는 어제 날짜를 기본값으로 설정
  const getDefaultDate = () => {
    if (period === 'daily') {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      return yesterday;
    }
    return new Date();
  };

  const [currentDate, setCurrentDate] = useState(getDefaultDate());

  // daily 모드일 때는 새로운 API 사용, 나머지는 기존 API 사용
  const {
    stats: dailyStats,
    loading: dailyLoading,
    error: dailyError,
    gptSummary,
    gptLoading,
    gptError
  } = useDailyStats(period === 'daily' ? currentDate : null);
  const { data: statisticsData, loading: statisticsLoading, error: statisticsError } = useStatistics(
    period !== 'daily' ? period : null,
    period !== 'daily' ? currentDate : null  // daily일 때는 null로 설정
  );

  // 현재 사용할 데이터 결정
  const data = period === 'daily' ? dailyStats : statisticsData;
  const loading = period === 'daily' ? dailyLoading : statisticsLoading;
  const error = period === 'daily' ? dailyError : statisticsError;



  // 오늘 날짜인지 확인 (daily 모드용)
  const isToday = period === 'daily' && (() => {
    const today = new Date();
    const selected = new Date(currentDate);
    return today.toDateString() === selected.toDateString();
  })();

  // 데이터가 비어있는지 확인 (daily 모드용)
  const isEmptyData = period === 'daily' && data && (
    data.totalCount === 0 &&
    data.insectTypeCount === 0 &&
    (!data.details || data.details.length === 0)
  );

  const farm = user?.selectedFarm;

  // 페이지 진입 시 body 스크롤 막기
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = 'unset';
      document.documentElement.style.overflow = 'unset';
    };
  }, []);

  if (!farm) {
    return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;
  }

  if (loading) {
    return <Loader message="통계 데이터를 불러오는 중입니다..." />;
  }

  if (error) {
    return <div className="text-center p-8 text-red-600">오류: {error}</div>;
  }

  // 기간별 제목 설정
  const getPeriodTitle = () => {
    switch(period) {
      case 'daily': return '일간 통계';
      case 'monthly': return '월간 통계';
      case 'yearly': return '연간 통계';
      default: return '통계';
    }
  };

  // 기간별 설명 설정
  const getPeriodDescription = () => {
    switch(period) {
      case 'daily': return '선택한 날짜의 해충 탐지 현황입니다.';
      case 'monthly': return '선택한 월의 해충 탐지 현황입니다.';
      case 'yearly': return '선택한 연도의 해충 탐지 현황입니다.';
      default: return '해충 탐지 현황입니다.';
    }
  };

  // 날짜 변경 핸들러
  const handleDateChange = (newDate) => {
    setCurrentDate(newDate);
  };

  return (
    <div className="section flex">
      {/* 왼쪽 패널 */}
      <LeftPanel />

      {/* 오른쪽 컨텐츠 영역 */}
      <div className="right-section flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          {/* 헤더 */}
          <div className="r-sec-top">
            <div>
              <p className="tit">{getPeriodTitle()}</p>
              <p className="desc">{getPeriodDescription()}</p>
            </div>
            {/* <button 
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button> */}
          </div>

        {/* 날짜 선택 */}
        <DateNavigation
          period={period}
          currentDate={currentDate}
          onDateChange={handleDateChange}
        />

        {/* 오늘 날짜 선택 시 메시지 */}
        {isToday && (
          <div className="mt-8 text-center py-12">
            <div className="text-gray-500 text-lg">
              📊 통계 준비중입니다
            </div>
            <div className="text-gray-400 text-sm mt-2">
              일간 통계는 하루가 완전히 끝난 후 확인할 수 있습니다
            </div>
          </div>
        )}

        {/* 빈 데이터 메시지 */}
        {!isToday && isEmptyData && (
          <div className="mt-8 text-center py-12">
            <div className="text-gray-500 text-lg">
              통계에 사용할 탐지 데이터가 없습니다
            </div>
          </div>
        )}

        {/* 데이터가 있을 때만 나머지 내용 표시 */}
        {!isToday && !isEmptyData && (
        <div
          key={`${period}-${currentDate.toISOString().split('T')[0]}`}
          className="report-content"
        >
            {/* gpt 분석 내용 */}
        <div className="baekgu-msg-wrap mt-8">
          <div className="thumb">
            <img src="/images/talk_109.png" alt="" />
          </div>
          <div className="baekgu-msg w-full">
            {period === 'daily' ? (
              gptLoading ? (
                '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.'
              ) : gptError ? (
                '분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
              ) : gptSummary ? (
                gptSummary
              ) : (
                '분석을 준비 중입니다.'
              )
            ) : (
              '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.'
            )}
          </div>
        </div>

        {/* 통계 내용 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* 총 탐지 수 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">총 탐지 수</h3>
            <div className="text-3xl font-bold text-[var(--color-primary)]">
              {period === 'daily' ? (data?.totalCount || 0) : (data?.totalDetections || 0)}마리
            </div>
          </div>

          {/* 탐지된 해충 종류 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">탐지된 해충 종류</h3>
            <div className="text-3xl font-bold text-[var(--color-accent)]">
              {period === 'daily' ? (data?.insectTypeCount || 0) : (data?.bugTypes || 0)}종
            </div>
          </div>

          {/* 가장 많이 탐지된 구역 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">최다 탐지 구역</h3>
            <div className="text-xl font-bold">
              {period === 'daily' ? (data?.topZone || '데이터 없음') : (data?.topRegion || '데이터 없음')}
            </div>
          </div>
        </div>

        {/* 차트 영역 (나중에 추가 가능) */}
        {/* <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">시간별 탐지 추이</h2>
          <div className="bordered-box h-64 flex items-center justify-center">
            <p className="text-gray-500">차트 영역 (추후 구현)</p>
          </div>
        </div> */}

        {/* 계절별 비교 (연간 통계에만 표시) */}
        {period === 'yearly' && (
          <div className="mt-8">
            <h2 className="text-xl font-bold mb-4">🌸 계절별 비교</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bordered-box text-center">
                <div className="text-2xl mb-2">🌸</div>
                <h3 className="font-bold mb-1">봄 (3-5월)</h3>
                <div className="text-xl font-bold text-green-600">123마리</div>
                <div className="text-sm text-gray-600">진딧물 多</div>
              </div>

              <div className="bordered-box text-center">
                <div className="text-2xl mb-2">☀️</div>
                <h3 className="font-bold mb-1">여름 (6-8월)</h3>
                <div className="text-xl font-bold text-red-600">456마리</div>
                <div className="text-sm text-gray-600">나방 多</div>
              </div>

              <div className="bordered-box text-center">
                <div className="text-2xl mb-2">🍂</div>
                <h3 className="font-bold mb-1">가을 (9-11월)</h3>
                <div className="text-xl font-bold text-orange-600">234마리</div>
                <div className="text-sm text-gray-600">거미 多</div>
              </div>

              <div className="bordered-box text-center">
                <div className="text-2xl mb-2">❄️</div>
                <h3 className="font-bold mb-1">겨울 (12-2월)</h3>
                <div className="text-xl font-bold text-blue-600">89마리</div>
                <div className="text-sm text-gray-600">활동 적음</div>
              </div>
            </div>
          </div>
        )}


        {/* 차트 영역 */}
        <div className="mt-8">
          <StatisticsChart data={data} period={period} />
        </div>

        {/* 상세 통계 - 토글 형태 */}
        <div className="mt-8">
          <GroupedDetailList data={data} period={period} />
        </div>

        </div>
        )}

        </div>
      </div>
    </div>
  );
}
