import React from "react";
import { useNavigate } from "react-router-dom";

export default function AlertList(){

	const navigate = useNavigate();

	return(
		<div className="alarm-area border border-[#666] p-4 shadow-sm rounded">
			<div className="flex justify-between items-center mb-3 text-xl">
				<div className="tit">확인하지 않은 알림</div>
				<div className="">
					<span className="text-red-600 font-bold text-[110%]">10</span> 건
				</div>
			</div>
			<ul className="alarm-list">
				{new Array(6).fill(null).map((_, i) => (
				<li
					key={idx}
					className="alarm-item border border-[#666] p-2 rounded">
					<div className="top flex items-center gap-2">
						<div className="al-bugName text-xl">노랑총채벌레</div>
						<div className="al-modelAcc bg-[var(--color-accent)] text-white px-1.5 py-0.5 rounded">95%</div>
					</div>
					<div className="btm flex items-center gap-2 mt-1">
						<div className="al-locatiaon rounded">A구역</div>
						<div className="al-time">2024년 7월 24일 14시 23분 30초</div>
					</div>
				</li>
				))}
			</ul>	
		</div>
	);
}