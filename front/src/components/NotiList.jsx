import React from "react";
import { useNotifications } from "../hooks/useNotifications";
import { useNavigate, useLocation } from "react-router-dom";

export default function NotiList(){

  const items = useNotifications();
	const count = items.length; // 알림 개수
  const navigate = useNavigate();
  const location = useLocation();

  // 현재 선택된 알림 ID 추출
  const getCurrentNotiId = () => {
    const match = location.pathname.match(/\/notifications\/(.+)/);
    return match ? match[1] : null;
  };

  const currentNotiId = getCurrentNotiId();

  const handleSelect = item => {
    navigate(`/notifications/${item.id}`);
  };	

	return(
		<div className="noti-area scrl-custom">
			<div className="noti-header">
				<div className="tit">확인하지 않은 알림</div>
				<div className="">
					<span className="noti-count">{count}</span> 건
				</div>
			</div>
			<ul className="noti-list">
				{items.map(item => (
				<li
					key={item.id}
					className={`noti-item hvborder ${currentNotiId === item.id.toString() ? 'active' : ''}`}
					onClick={() => handleSelect(item)}
				>
					<div className="noti-item-top">
						<div className="noti-bug-name">{item.bugName}</div>
						<div className="text-sm">(신뢰도 {item.accuracy}%)</div> {/*noti-accuracy*/}
					</div>
					<div className="noti-item-bottom text-sm">
						<div className="">{item.location}&nbsp;&nbsp;{item.timestamp}</div>
					</div>
				</li>
				))}
			</ul>
		</div>
	);
}