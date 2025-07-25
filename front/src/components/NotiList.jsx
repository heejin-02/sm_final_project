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
		<div className="noti-area border border-[#666] p-4 shadow-sm rounded overflow-y-auto flex-0-auto">
			<div className="flex justify-between items-center mb-3 text-xl">
				<div className="tit">확인하지 않은 알림</div>
				<div className="">
					<span className="text-red-600 font-bold text-[110%]">{count}</span> 건
				</div>
			</div>
			<ul className="noti-list space-y-2">
				{items.map(item => (
				<li
					key={item.id}
					className="noti-item border border-[#666] p-2 rounded"
					onClick={() => handleSelect(item)}
				>
					<div className="top flex items-center gap-1">
						<div className="noti-bugName text-lg">{item.bugName}</div>
						<div className="noti-modelAcc bg-[var(--color-accent)] text-white px-1.5 py-0.5 rounded">{item.accuracy}%</div>
					</div>
					<div className="btm flex items-center gap-2 mt-1">
						<div className="">{item.location} / {item.timestamp}</div>
					</div>
				</li>
				))}
			</ul>	
		</div>
	);
}