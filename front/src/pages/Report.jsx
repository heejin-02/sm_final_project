// src/pages/Report.jsx
import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useStatistics } from '../hooks/useStatistics';
import Loader from '../components/Loader';
import LeftPanel from '../components/LeftPanel';
import DateNavigation from '../components/DateNavigation';
import GroupedDetailList from '../components/GroupedDetailList';

export default function Report() {
  const { period } = useParams(); // 'daily', 'monthly', 'yearly'
  const { user } = useAuth();
  const navigate = useNavigate();
  const [currentDate, setCurrentDate] = useState(new Date());
  const { data, loading, error } = useStatistics(period, currentDate);

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
    // TODO: 새로운 날짜로 데이터 다시 로드
    console.log('날짜 변경:', newDate);
  };

  return (
    <div className="flex h-screen">
      {/* 왼쪽 패널 */}
      <LeftPanel />

      {/* 오른쪽 컨텐츠 영역 */}
      <div className="right-section flex-1 overflow-y-auto p-4">
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

        {/* 통계 내용 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* 총 탐지 수 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">총 탐지 수</h3>
            <div className="text-3xl font-bold text-[var(--color-primary)]">
              {data?.totalDetections || 0}마리
            </div>
          </div>

          {/* 탐지된 해충 종류 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">탐지된 해충 종류</h3>
            <div className="text-3xl font-bold text-[var(--color-accent)]">
              {data?.bugTypes || 0}종
            </div>
          </div>

          {/* 가장 많이 탐지된 구역 */}
          <div className="bordered-box">
            <h3 className="font-bold mb-2">최다 탐지 구역</h3>
            <div className="text-xl font-bold">
              {data?.topRegion || '데이터 없음'}
            </div>
          </div>
        </div>

        {/* 상세 통계 - 토글 형태 */}
        <div className="mt-8">
          <GroupedDetailList data={data} period={period} />
        </div>

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

        {/* 차트 영역 (나중에 추가 가능) */}
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">시간별 탐지 추이</h2>
          <div className="bordered-box h-64 flex items-center justify-center">
            <p className="text-gray-500">차트 영역 (추후 구현)</p>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}
