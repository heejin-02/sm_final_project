import React, { useEffect, useState } from "react";
import { useNotifications } from "../hooks/useNotifications";
import { useNavigate } from "react-router-dom";

export default function NotiList(){

  const items = useNotifications();
	const count = items.length; // 알림 개수
  const navigate = useNavigate();

  const handleSelect = item => {
    navigate(`/notifications/${item.id}`);
  };	

	return(
		<div className="noti-area">
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
					className="noti-item"
					onClick={() => handleSelect(item)}
				>
					<div className="noti-item-top">
						<div className="noti-bug-name">{item.bugName}</div>
						<div className="noti-accuracy">{item.accuracy}%</div>
					</div>
					<div className="noti-item-bottom">
						<div className="">{item.location} / {item.timestamp}</div>
					</div>
				</li>
				))}
			</ul>
		</div>
	);
}