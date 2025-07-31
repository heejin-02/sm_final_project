import React from "react";
import { useAlertList } from "../hooks/useAlerts";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import Loader from "./Loader";

export default function NotiList(){

  const { user } = useAuth();
  const farmIdx = user?.selectedFarm?.farmIdx;

  const { alerts, loading, error, markAsRead } = useAlertList(farmIdx);
  const unreadCount = alerts.filter(alert => !alert.isRead).length;

  const navigate = useNavigate();
  const location = useLocation();

  // 현재 선택된 알림 ID 추출
  const getCurrentNotiId = () => {
    const match = location.pathname.match(/\/notifications\/(.+)/);
    return match ? parseInt(match[1]) : null;
  };

  const currentNotiId = getCurrentNotiId();

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
		<div className="noti-area scrl-custom">
			<div className="noti-header">
				<div className="tit">오늘의 알림</div>
				<div className="text-sm">
					미확인 <span className="noti-count">{unreadCount}</span> 건
				</div>
			</div>
			<ul className="noti-list">
				{alerts.length === 0 ? (
					<li className="noti-item text-center text-gray-500 py-8">
						알림이 없습니다.
					</li>
				) : (
					alerts.map((alert, index) => (
						<li
							key={alert.anlsIdx}
							className={`noti-item hvborder ${currentNotiId === alert.anlsIdx ? 'active' : ''} ${!alert.isRead ? 'unread' : ''}`}
							onClick={() => handleSelect(alert)}
						>
							<div className="noti-item-top">
								<div className="noti-bug-name">{alert.insectName}</div>
								<div className="text-sm">
									(신뢰도 {alert.anlsAcc}%)
									{!alert.isRead && <span className="ml-2 text-red-500">●</span>}
								</div>
							</div>
							<div className="noti-item-bottom text-sm">
								<div className="">{alert.gh_name || `${index + 1}번 구역`}&nbsp;&nbsp;{alert.anlsDate}</div>
							</div>
						</li>
					))
				)}
			</ul>
		</div>
	);
}