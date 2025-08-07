// src/components/GroupedDetailList.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function GroupedDetailList({ data, period }) {
  const [expandedGroups, setExpandedGroups] = useState(new Set());
  const navigate = useNavigate();

  // 그룹 토글
  const toggleGroup = (groupKey) => {
    const newExpanded = new Set(expandedGroups);
    if (newExpanded.has(groupKey)) {
      newExpanded.delete(groupKey);
    } else {
      newExpanded.add(groupKey);
    }
    setExpandedGroups(newExpanded);
  };

  // 행 클릭 시 NotiDetail로 이동
  const handleRowClick = (anlsIdx) => {
    navigate(`/noti-detail/${anlsIdx}`);
  };

  // 일간은 기존 방식 (그룹핑 없음)
  if (period === 'daily') {
    return (
      <div className="bordered-box">
        <h2 className="text-xl font-bold mb-4">상세 현황</h2>
        <div className="table-wrap scrl-custom">
          <table className="table border">
            <thead>
              <tr className="border-b">
                <th className="">탐지 시간</th>
                <th className="">탐지 구역</th>
                <th className="">해충 이름</th>
                <th className="">탐지 정확도</th>
              </tr>
            </thead>
            <tbody>
              {data?.details?.length > 0 ? (
                data.details.map((item, index) => (
                  <tr
                    key={index}
                    className="border-b hover:bg-gray-50 cursor-pointer"
                    onClick={() => handleRowClick(item.anlsIdx || index)}
                  >
                    <td className="p-3">{item.time}</td>
                    <td className="p-3">{item.greenhouse}</td>
                    <td className="p-3">{item.insect}</td>
                    <td className="p-3">{item.accuracy}%</td>
                  </tr>
                ))
              ) : data?.detailList?.length > 0 ? (
                // 기존 구조 fallback
                data.detailList.map((item) => (
                  <tr
                    key={item.anlsIdx}
                    className="border-b hover:bg-gray-50 cursor-pointer"
                    onClick={() => handleRowClick(item.anlsIdx)}
                  >
                    <td className="p-3">{item.datetime}</td>
                    <td className="p-3">{item.region}</td>
                    <td className="p-3">{item.bugType}</td>
                    <td className="p-3">{item.accuracy}%</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" className="p-4 text-center text-gray-500">
                    데이터가 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // 월간/연간은 그룹핑된 토글 방식
  return (
    <div className="bordered-box grouped-detail-list">
      <h2 className="text-xl font-bold mb-4">상세 현황</h2>

      {/* 월간 데이터의 새로운 구조 처리 */}
      {period === 'monthly' && data?.details ? (
        <div className="table-wrap scrl-custom">
          <table className="table border">
            <thead>
              <tr className="border-b">
                <th className="">탐지 시간</th>
                <th className="">탐지 구역</th>
                <th className="">해충 이름</th>
                <th className="">탐지 정확도</th>
              </tr>
            </thead>
            <tbody>
              {data.details.map((item, index) => (
                <tr
                  key={index}
                  className="border-b hover:bg-gray-50 cursor-pointer"
                  onClick={() => handleRowClick(item.anlsIdx || index)}
                >
                  <td className="p-3">{item.time}</td>
                  <td className="p-3">{item.greenhouse}</td>
                  <td className="p-3">{item.insect}</td>
                  <td className="p-3">{item.accuracy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : data?.groupedData ? (
        <div className="space-y-2">
          {Object.entries(data.groupedData).map(([groupKey, group]) => (
            <div key={groupKey} className="border rounded-lg overflow-hidden">
              {/* 그룹 헤더 (토글 버튼) */}
              <button
                className="w-full p-4 text-left hover:bg-gray-50 flex items-center justify-between"
                onClick={() => toggleGroup(groupKey)}
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">
                    {expandedGroups.has(groupKey) ? '▼' : '▶'}
                  </span>
                  <span className="font-semibold">{group.title}</span>
                  <span className="text-sm text-gray-600">({group.count}건)</span>
                </div>
              </button>

              {/* 그룹 내용 (토글 시 표시) */}
              {expandedGroups.has(groupKey) && (
                <div className="table-wrap">
                  <table className="table border">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="">탐지 시간</th>
                        <th className="">탐지 구역</th>
                        <th className="">해충 이름</th>
                        <th className="">탐지 정확도</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.items.map((item) => (
                        <tr
                          key={item.anlsIdx}
                          className="border-b hover:bg-blue-50 cursor-pointer"
                          onClick={() => handleRowClick(item.anlsIdx)}
                        >
                          <td className="">{item.datetime}</td>
                          <td className="">{item.region}</td>
                          <td className="">{item.bugType}</td>
                          <td className="">{item.accuracy}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="p-4 text-center text-gray-500">
          데이터가 없습니다.
        </div>
      )}
    </div>
  );
}
