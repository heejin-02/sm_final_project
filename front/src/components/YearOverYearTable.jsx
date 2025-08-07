// src/components/YearOverYearTable.jsx
import React from 'react';

export default function YearOverYearTable({
  months,        // ['1월', '2월', …, '12월']
  previousYear,  // [12, 24, …]
  currentYear,   // [14, 23, …]
  nextYear,      // [16, 22, …] — 예측치
  highlightUpTo, // number, 현재 선택된 월 (예: 8)
}) {
  return (
		<div className="bordered-box mt-8 year-over-year">
			<h3 className="text-lg font-bold mb-4">내년 해충 발생 예측</h3>
			<div className="overflow-x-auto mt-8">
				<table className="table">
					<thead>
						<tr>
							<th className="">년도</th>
							{months.map((m, i) => (
								<th key={i} className="">
									{m}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{[
							{ label: `${new Date().getFullYear()-1}`, data: previousYear },
							{ label: `${new Date().getFullYear()}`,     data: currentYear },
							{ label: '내년(예측)',                       data: nextYear }
						].map(({ label, data }) => (
							<tr key={label} className="">
								<td className="">{label}</td>
								{months.map((_, idx) => (
									<td
										key={idx}
										className={`${
											idx >= highlightUpTo ? 'text-gray-400' : ''
										}`}
									>
										{data[idx] != null ? data[idx] : '-'}
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</div>	

  );
}
