// src/pages/Report.jsx
import { useState, useEffect } from 'react';
import { useParams }       from 'react-router-dom';
import { useAuth }         from '../contexts/AuthContext';
import { useStatistics }   from '../hooks/useStatistics';
import Loader              from '../components/Loader';
import LeftPanel           from '../components/LeftPanel';
import DateNavigation      from '../components/DateNavigation';
import GroupedDetailList   from '../components/GroupedDetailList';
import StatisticsChart     from '../components/StatisticsChart';

// 새로 만든 테이블 컴포넌트
import YearOverYearTable from '../components/YearOverYearTable';

export default function Report() {
  const { period } = useParams(); // 'daily' | 'monthly' | 'yearly'
  const { user }   = useAuth();

  // 기본 날짜 설정
  const getDefaultDate = () => {
    const d = new Date();
    if (period === 'daily') d.setDate(d.getDate() - 1);
    return d;
  };

  const [currentDate, setCurrentDate] = useState(() => getDefaultDate());
  const {
    stats,
    loading,
    error,
    gptSummary,
    gptLoading,
    gptError,
    refetch
  } = useStatistics({ period, date: currentDate });

  // 일간, 월간, 연간 클릭할 때마다 오늘 날짜 기준으로 초기화
  useEffect(() => {
    setCurrentDate(getDefaultDate());
  }, [period]);

  // daily 전용: 오늘인지 체크
  const isToday = period === 'daily'
    && new Date().toDateString() === new Date(currentDate).toDateString();

  // daily 전용: 빈 데이터 여부
  const isEmptyData = period === 'daily'
    && stats
    && stats.totalCount === 0
    && stats.insectTypeCount === 0
    && (!stats.details || stats.details.length === 0);

  // 진입 시 스크롤 잠금
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
      document.documentElement.style.overflow = '';
    };
  }, []);

  if (!user?.selectedFarm) {
    return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;
  }
  if (loading) {
    return <Loader message="통계 데이터를 불러오는 중입니다..." />;
  }
  if (error) {
    return <div className="text-center p-8 text-red-600">오류: {error}</div>;
  }

  // stats가 아직 null이면, 데이터 준비 중이므로 로더만 보여주기
  if (!stats) {
   return <Loader message="통계 데이터를 불러오는 중입니다..." />;
  }

  // 제목·설명 헬퍼
  const getPeriodTitle = () => ({
    daily:   '일간 통계',
    monthly: '월간 통계',
    yearly:  '연간 통계'
  }[period] || '통계');

  const getPeriodDescription = () => ({
    daily:   '선택한 날짜의 해충 탐지 현황입니다.',
    monthly: '선택한 월의 해충 탐지 현황입니다.',
    yearly:  '선택한 연도의 해충 탐지 현황입니다.'
  }[period] || '해충 탐지 현황입니다.');

  // 날짜 변경 핸들러
  const handleDateChange = newDate => {
    setCurrentDate(newDate);
  };

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

          {/* 하루 통계 준비 중 메시지 */}
          {isToday && (
            <div className="mt-8 text-center py-12">
              <p className="text-gray-500 text-lg">📊 통계 준비중입니다</p>
              <p className="text-gray-400 text-sm mt-2">
                일간 통계는 하루가 완전히 끝난 후 확인할 수 있습니다
              </p>
            </div>
          )}

          {/* 빈 데이터 메시지 */}
          {!isToday && isEmptyData && (
            <div className="mt-8 text-center py-12">
              <p className="text-gray-500 text-lg">통계에 사용할 탐지 데이터가 없습니다</p>
            </div>
          )}

          {/* 실제 데이터 */}
          {!isToday && !isEmptyData && (
            <div
              key={`${period}-${currentDate.toISOString().split('T')[0]}`}
              className="report-content"
            >
              {/* GPT 분석 */}
              <div className="baekgu-msg-wrap mt-8 flex">
                <div className="thumb mr-4">
                  <img src="/images/talk_109.png" alt="백구" />
                </div>
                <div className="baekgu-msg w-full">
                  {period === 'daily' ? (
                    gptLoading   ? '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.' :
                    gptError     ? '분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' :
                    gptSummary   || '분석을 준비 중입니다.'
                  ) : period === 'monthly' ? (
                    gptLoading   ? '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.' :
                    gptError     ? '분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' :
                    gptSummary   || '분석을 준비 중입니다.'
                  ) : (
                    '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.'
                  )}
                </div>
              </div>

              {/* 요약 카드 */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">총 탐지 수</h3>
                  <p className="text-3xl font-bold">
                    {stats?.totalCount ?? 0}마리
                  </p>
                </div>
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">탐지된 해충 종류</h3>
                  <p className="text-3xl font-bold">
                    {stats?.insectTypeCount ?? 0}종
                  </p>
                </div>
                <div className="bordered-box">
                  <h3 className="font-bold mb-2">최다 탐지 구역</h3>
                  <p className="text-xl font-bold">
                    {stats?.topZone ?? '데이터 없음'}
                  </p>
                </div>
              </div>

              {/* 연간: 계절별 비교 */}
              {period === 'yearly' && stats.seasonal && (
                <div className="mt-8">
                  <h2 className="text-xl font-bold mb-4">🌸 계절별 비교</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {stats.seasonal.map(({ season, count, topBug }) => (
                      <div key={season} className="bordered-box text-center">
                        <p className="text-2xl mb-2">{season.emoji}</p>
                        <h3 className="font-bold mb-1">{season.label}</h3>
                        <p className="text-xl font-bold">{count}마리</p>
                        <p className="text-sm text-gray-600">{topBug} 多</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 테이블 (더미 데이터 테스트용) */}
              {period === 'monthly' && (
                <YearOverYearTable
                  // 보여줄 월 레이블 배열
                  months={['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월','9월', '10월', '11월', '12월']}
                  // 전년도 월별 탐지 수 (배열)
                  previousYear={[12, 24, 45, 12, 35, 25, 35, 32, 15, 26, 12, 25]}
                  // 올해 월별 탐지 수 (배열)
                  currentYear={[14, 23, 52, 23, 23, 75, 43, 32]}
                  // 내년 예측치 (배열)
                  nextYear={[16, 22, 60, 32, 16, 99, 53, 32]}
                  // 강조(진하게)할 마지막 인덱스 (여기선 8월까지 → 8)
                  highlightUpTo={8}
                />
              )}

              {period === 'yearly' && (
                <YearOverYearTable
                  // 보여줄 월 레이블 배열
                  months={['봄(3~5월)', '여름(6~8월)', '가을(9~11월)', '겨울(12~2월)']}
                  // 전년도 월별 탐지 수 (배열)
                  previousYear={[125, 245, 455, 125]}
                  // 올해 월별 탐지 수 (배열)
                  currentYear={[145, 235]}
                  // 내년 예측치 (배열)
                  nextYear={[165, 225]}
                  // 강조(진하게)할 마지막 인덱스 (여기선 8월까지 → 8)
                  highlightUpTo={2}
                />
              )}              

              {/* 차트 */}
              <div className="mt-8">
                <StatisticsChart data={stats} period={period} />
              </div>

              {/* 상세 리스트 */}
              <div className="mt-8">
                <GroupedDetailList data={stats} period={period} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
