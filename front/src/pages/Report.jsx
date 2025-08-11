import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useStatistics } from '../hooks/useStatistics';
import Loader from '../components/Loader';
import LeftPanel from '../components/LeftPanel';
import DateNavigation from '../components/DateNavigation';
import GroupedDetailList from '../components/GroupedDetailList';
import StatisticsChart from '../components/StatisticsChart';
import MonthlyTable from '../components/MonthlyTable'; 
import YearOverYearTable from '../components/YearOverYearTable';
import GptSummary from '../components/GptSummary'; 

export default function Report() {
  const { period } = useParams(); // 'daily' | 'monthly' | 'yearly'
  const { user } = useAuth();

  const getDefaultDate = () => {
    const d = new Date();
    if (period === 'daily') d.setDate(d.getDate() - 1);
    return d;
  };

  const [currentDate, setCurrentDate] = useState(() => getDefaultDate());

  const { stats, loading, error, refetch }   = useStatistics({ period, date: currentDate });

  useEffect(() => {
    setCurrentDate(getDefaultDate());
  }, [period]);

  useEffect(() => {
    document.body.style.overflowY = 'hidden';
    document.documentElement.style.overflowY = 'hidden';
    return () => {
      document.body.style.overflowY = '';
      document.documentElement.style.overflowY = '';
    };
  }, []);

  const isToday = period === 'daily'
    && new Date().toDateString() === new Date(currentDate).toDateString();

  const isEmptyData = period === 'daily'
    && stats
    && stats.totalCount === 0
    && stats.insectTypeCount === 0
    && (!stats.details || stats.details.length === 0);

  const getPeriodTitle = () => ({
    daily: '일간 통계',
    monthly: '월간 통계',
    yearly: '연간 통계'
  }[period] || '통계');

  const getPeriodDescription = () => ({
    daily: '선택한 날짜의 해충 탐지 현황입니다.',
    monthly: '선택한 월의 해충 탐지 현황입니다.',
    yearly: '선택한 연도의 해충 탐지 현황입니다.'
  }[period] || '해충 탐지 현황입니다.');

  const handleDateChange = newDate => {
    setCurrentDate(newDate);
  };

  const year = currentDate.getFullYear();
  const month = String(currentDate.getMonth() + 1).padStart(2, '0');

  const colLabels = [
    `${year - 1}.${month}`,
    `${year}.${month}`,
    `${year + 1}.${month} 예측`
  ];

  const types = stats?.predictions?.map(p => p.insectName) ?? [];

  const matrix = stats?.predictions?.map(p => [
    p.count2024,
    p.count2025,
    p.predicted2026
  ]) ?? [];

  if (!user?.selectedFarm) {
    return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;
  }

  if (loading) {
    return <Loader message="통계 데이터를 불러오는 중입니다..." />;
  }

  if (error) {
    return <div className="text-center p-8 text-red-600">오류: {error}</div>;
  }

  if (!stats) {
    return <Loader message="통계 데이터를 불러오는 중입니다..." />;
  }

  return (
    <div className="section flex">
      <LeftPanel />

      <div className="right-section flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          {/* 헤더 */}
          <div className="r-sec-top mb-6 flex justify-between items-center">
            <div>
              <p className="tit">{getPeriodTitle()}</p>
              <p className="desc">{getPeriodDescription()}</p>
            </div>
          </div>

          {/* 날짜 선택 */}
          <DateNavigation
            period={period}
            currentDate={currentDate}
            onDateChange={handleDateChange}
          />

          {isToday && (
            <div className="mt-8 text-center py-12">
              <p className="text-gray-500 text-lg">📊 통계 준비중입니다</p>
              <p className="text-gray-400 text-sm mt-2">
                일간 통계는 하루가 완전히 끝난 후 확인할 수 있습니다
              </p>
            </div>
          )}

          {!isToday && isEmptyData && (
            <div className="mt-8 text-center py-12">
              <p className="text-gray-500 text-lg">통계에 사용할 탐지 데이터가 없습니다</p>
            </div>
          )}

          {!isToday && !isEmptyData && (
            <div
              key={`${period}-${currentDate.toISOString().split('T')[0]}`}
              className="report-content"
            >
              {/* GPT 분석 */}
              <GptSummary period={period} date={currentDate} />

              {/* 요약 카드 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">총 탐지 수</h3>
                  <p className="text-3xl font-bold">{stats?.totalCount ?? 0}마리</p>
                </div>
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">탐지된 해충 종류</h3>
                  <p className="text-3xl font-bold">{stats?.insectTypeCount ?? 0}종</p>
                </div>
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">최다 탐지 구역</h3>
                  <p className="text-xl font-bold">{stats?.topZone ?? '데이터 없음'}</p>
                </div>
              </div>

              {/* 내년 예측 테이블 */}
              <div className="mt-8 predicted-table">
                {/* 월간 통계 */}
                {period === 'monthly' && stats && <MonthlyTable predictions={stats?.predictions} />}
                
                {/* 연간 통계 */}
                {period === 'yearly' && stats && <YearOverYearTable stats={stats} />}
              </div>

              {/* 차트 */}
              <div className="mt-8">
                <StatisticsChart stats={stats} period={period} currentDate={currentDate} />
              </div>

              {/* 상세 현황 */}
              <div className="mt-8">
                <GroupedDetailList stats={stats} period={period} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
