// src/pages/NotiDetail.jsx
import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import LeftPanel from '../components/LeftPanel';
import { useNotifications } from '../hooks/useNotifications';
import { useRegions } from '../hooks/useRegions';
import NotiFarmMap from '../components/NotiFarmMap';
import DetectionFeedback from '../components/DetectionFeedback';



export default function NotiDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const notifications = useNotifications();
  const { regions, loading: regionsLoading } = useRegions();

  // 페이지 진입 시 body 스크롤 막기
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = 'unset';
      document.documentElement.style.overflow = 'unset';
    };
  }, []);

  // 현재 알림 찾기
  const notification = notifications.find(item => item.id.toString() === id);

  // 피드백 제출 핸들러
  const handleFeedbackSubmit = (feedbackData) => {
    console.log('피드백 데이터:', feedbackData);
    // TODO: 실제 API로 피드백 전송
    // 중복 팝업 제거 - DetectionFeedback 컴포넌트에서 이미 완료 메시지 표시
  };

  // 나중에 확인하기 핸들러
  const handleMarkAsRead = (notificationId) => {
    console.log('알림 확인 처리:', notificationId);
    // TODO: 실제 API로 알림 읽음 처리
    alert('알림이 확인 처리되었습니다.');
    // 이전 페이지로 이동
    navigate(-1);
  };

  // 알림을 찾지 못한 경우
  if (!notification) {
    return (
      <div className="flex h-screen">
        <LeftPanel />
        <div className="right-section">
          <div className="text-center p-8">
            <h1 className="text-2xl font-bold mb-4">알림을 찾을 수 없습니다</h1>
            <p className="text-gray-600 mb-4">요청하신 알림이 존재하지 않거나 삭제되었습니다.</p>
            {/* <button
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button> */}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* 왼쪽 패널 */}
      <LeftPanel />
      
      {/* 오른쪽 컨텐츠 영역 */}
      <div className="right-section space-y-6">
          {/* 헤더 */}
          <div className="r-sec-top">
            <div>
              <p className="tit">해충 탐지 알림 내용</p>
              <p className="desc">탐지된 해충에 대한 상세 정보입니다.</p>
            </div>
            {/* <button 
              onClick={() => navigate(-1)}
              className="btn-submit px-4 py-2 text-sm"
            >
              뒤로가기
            </button> */}
          </div>

          {/* 탐지 영상 */}
          <div className="flex gap-4">
            <div className="bordered-box flex-1/2">
              {/* <h3 className="tit-2 text-center">탐지 구역</h3> */}
              <div className="text-center mb-3">
                <span className="text-gray-600 text-lg">
                  <span className="font-semibold text-black">{notification.location}</span> 에서&nbsp;
                  <span className="font-semibold text-black">{notification.bugName}</span> 탐지됨&nbsp; 
                  <span className='text-base'>(신뢰도 {notification.accuracy}%)</span>
                </span>
                <div>{notification.timestamp}</div>
              </div>              
              <NotiFarmMap
                highlightRegion={notification.location}
                regions={regions}
                loading={regionsLoading}
              />
            </div>            
            <div className="bordered-box flex-1/2">
              <h3 className="tit-2 text-center">탐지 영상</h3>
              <div className="video_wrap">
                <video src="http://192.168.219.72:8095/videos/20250725/2_20250725_113404.mp4" controls/>
              </div>
            </div>
          </div>



          {/* AI 탐지 결과 피드백 */}
          <DetectionFeedback
            notification={notification}
            onFeedbackSubmit={handleFeedbackSubmit}
            onMarkAsRead={handleMarkAsRead}
          />

      </div>
    </div>
  );
}
