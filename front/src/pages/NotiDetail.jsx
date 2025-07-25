// src/pages/NotiDetail.jsx
import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import LeftPanel from '../components/LeftPanel';

export default function NotiDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  // 페이지 진입 시 body 스크롤 막기
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = 'unset';
      document.documentElement.style.overflow = 'unset';
    };
  }, []);

  // TODO: 실제 알림 데이터 가져오기
  const notification = {
    id: id,
    bugName: '총채벌레',
    accuracy: 98,
    location: 'A구역',
    timestamp: '2024년 01월 15일 14시 23분 30초',
    videoUrl: '/videos/detection_sample.mp4', // 임시
    description: '선택된 알림에 대한 자세한 알림 내용'
  };

  return (
    <div className="flex h-screen">
      {/* 왼쪽 패널 */}
      <LeftPanel />
      
      {/* 오른쪽 컨텐츠 영역 */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="tit mb-2">알림 상세</h1>
              <p className="text-gray-600">탐지된 해충에 대한 상세 정보입니다.</p>
            </div>
            <button 
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button>
          </div>

          {/* 알림 정보 카드 */}
          <div className="bordered-box mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">{notification.timestamp}</h2>
              <div className="flex items-center gap-2">
                <span className="text-lg font-semibold">{notification.location}</span>
                <span className="noti-accuracy">{notification.accuracy}%</span>
              </div>
            </div>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl font-bold text-[var(--color-primary)]">
                {notification.bugName}
              </span>
              <span className="text-gray-600">탐지됨 (신뢰도 {notification.accuracy}%)</span>
            </div>
            
            <p className="text-gray-700">{notification.description}</p>
          </div>

          {/* 탐지 영상 */}
          <div className="bordered-box mb-6">
            <h3 className="text-lg font-bold mb-4">탐지 영상</h3>
            <div className="bg-gray-200 rounded-lg aspect-video flex items-center justify-center">
              {/* TODO: 실제 비디오 플레이어로 교체 */}
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-400 rounded-full flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-gray-600">벌레 탐지 영상(10초간)</p>
              </div>
            </div>
          </div>

          {/* 탐지 구역 정보 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bordered-box">
              <h3 className="font-bold mb-2">탐지 구역</h3>
              <div className="text-2xl font-bold text-[var(--color-accent)]">
                {notification.location}
              </div>
              <p className="text-sm text-gray-600 mt-1">박용필농장</p>
            </div>
            
            <div className="bordered-box">
              <h3 className="font-bold mb-2">오늘 찾은 해충</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>오전 10:02:30 A구역(문앞)</span>
                  <span className="noti-accuracy">98%</span>
                </div>
                <div className="flex justify-between">
                  <span>오전 11:02:30 A구역(문앞)</span>
                  <span className="noti-accuracy">98%</span>
                </div>
                <div className="flex justify-between">
                  <span>오후 12:02:30 A구역(문앞)</span>
                  <span className="noti-accuracy">98%</span>
                </div>
              </div>
            </div>
          </div>

          {/* 발견 구역 통계 */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bordered-box text-center">
              <h4 className="font-bold mb-2">해충 발견</h4>
              <div className="text-2xl font-bold text-[var(--color-primary)]">5</div>
              <div className="text-sm text-gray-600">마리</div>
            </div>
            
            <div className="bordered-box text-center">
              <h4 className="font-bold mb-2">발견 구역</h4>
              <div className="text-2xl font-bold text-[var(--color-accent)]">2</div>
              <div className="text-sm text-gray-600">곳</div>
            </div>
            
            <div className="bordered-box text-center">
              <h4 className="font-bold mb-2">발견 종류</h4>
              <div className="text-2xl font-bold text-[var(--color-yellow)]">2</div>
              <div className="text-sm text-gray-600">종</div>
            </div>
          </div>

          {/* 백구의 하루 */}
          <div className="bordered-box">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              백구의 하루 🐕
            </h3>
            <p className="text-gray-700 leading-relaxed">
              오늘은 어제 보다 벌레가 없네요! 다만 B 구역에서만 4 마리의 벌레가 발견 되어 확인이 필요 합니다!
              다음 주에 비가 내릴 예정 입니다. 습도 관리에 신경써 주세요. 백구가 계속 지켜 볼게요!
            </p>
          </div>

          {/* 액션 버튼들 */}
          <div className="flex gap-4 mt-6">
            <button className="flex-1 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600">
              백구에게 답장하기
            </button>
            <button className="flex-1 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:opacity-90">
              나중에 답장하기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
