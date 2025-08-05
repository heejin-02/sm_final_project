import React, { useEffect } from "react";
import { useAlertList } from "../hooks/useAlerts";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import Loader from "./Loader";

export default function NotiList(){

  const { user } = useAuth();
  const farmIdx = user?.selectedFarm?.farmIdx;

  const { alerts, loading, error, markAsRead } = useAlertList(farmIdx);

  // 알림 정렬: 안읽은 알림 우선 + 각각 최신순
  const sortedAlerts = [...alerts].sort((a, b) => {
    // 1. 읽음 상태로 먼저 정렬 (안읽은 것이 위로)
    if (a.notiCheck !== b.notiCheck) {
      if (a.notiCheck === "N" && b.notiCheck === "Y") return -1; // a가 위로
      if (a.notiCheck === "Y" && b.notiCheck === "N") return 1;  // b가 위로
    }
    // 2. 같은 읽음 상태 내에서는 anlsIdx 큰 순 (최신순)
    return b.anlsIdx - a.anlsIdx;
  });

  const unreadCount = alerts.filter(alert => alert.notiCheck !== "Y").length;

  const navigate = useNavigate();
  const location = useLocation();

  // 현재 선택된 알림 ID 추출
  const getCurrentNotiId = () => {
    const match = location.pathname.match(/\/notifications\/(.+)/);
    return match ? parseInt(match[1]) : null;
  };

  const currentNotiId = getCurrentNotiId();

  // 현재 선택된 알림으로 스크롤 (noti-list 컨테이너 내에서만)
  useEffect(() => {
    if (currentNotiId) {
      const element = document.querySelector(`[data-anls-idx="${currentNotiId}"]`);
      const container = document.querySelector('.noti-list');

      if (element && container) {
        // 컨테이너 내에서의 상대적 위치 계산
        const scrollTop = element.offsetTop - container.offsetTop;

        // noti-list 컨테이너만 스크롤
        container.scrollTo({
          top: scrollTop,
          behavior: 'smooth'
        });
      }
    }
  }, [currentNotiId, sortedAlerts]); // sortedAlerts가 변경될 때도 스크롤

  const handleSelect = async (alert) => {
    // 읽음 처리
    await markAsRead(alert.anlsIdx);
    // 상세 페이지로 이동
    navigate(`/notifications/${alert.anlsIdx}`);
  };

  // 로딩 중이거나 농장 정보가 없으면 로딩 표시
  if (loading || !farmIdx) {
    return (
      <div className="noti-area">
        <div className="noti-header">
          <div className="tit">오늘의 알림</div>
          <div className="text-sm">로딩 중...</div>
        </div>
        <div className="flex justify-center items-center h-32">
          <Loader />
        </div>
      </div>
    );
  }

  // 에러 처리
  if (error) {
    return (
      <div className="noti-area">
        <div className="noti-header">
          <div className="tit">오늘의 알림</div>
          <div className="text-sm text-red-500">오류 발생</div>
        </div>
        <div className="p-4 text-center text-red-500">
          알림을 불러오는데 실패했습니다.
        </div>
      </div>
    );
  }

	return(
		<div className="noti-area">
			<div className="noti-header">
				<div className="tit">오늘의 알림</div>
				<div className="text-sm">
					미확인 <span className="noti-count">{unreadCount}</span> 건
				</div>
			</div>
			<ul className="noti-list scrl-custom">
				{sortedAlerts.length === 0 ? (
					<li className="noti-item text-center text-gray-500 py-8">
						알림이 없습니다.
					</li>
				) : (
					sortedAlerts.map((alert, index) => (
						<li
							key={alert.anlsIdx}
							className={`noti-item ${currentNotiId === alert.anlsIdx ? 'active' : ''} ${alert.notiCheck !== "Y" ? 'unread' : ''}`}
							onClick={() => handleSelect(alert)}
              data-anls-idx={alert.anlsIdx}
						>
							<div className="noti-item-top">
								<div className="noti-bug-name">{alert.insectName}</div>
								<div className="noti-acc">(신뢰도 {alert.anlsAcc}%)</div>
                {alert.notiCheck !== "Y" && <span className="red-dot"></span>}
							</div>
							<div className="noti-item-bottom">
								<div className="">{alert.ghName}&nbsp;&nbsp;{alert.createdAt}</div>
							</div>
						</li>
					))
				)}
			</ul>
		</div>
	);
}