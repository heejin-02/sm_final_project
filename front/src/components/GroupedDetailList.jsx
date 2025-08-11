// src/components/GroupedDetailList.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { groupDataByWeek } from '../utils/groupDataByWeek';
import { groupDataByMonth } from '../utils/groupDataByMonth';

export default function GroupedDetailList({ stats, period }) {
  const [expandedGroups, setExpandedGroups] = useState(new Set());
  const navigate = useNavigate();

  const toggleGroup = (groupKey) => {
    const newExpanded = new Set(expandedGroups);
    newExpanded.has(groupKey) ? newExpanded.delete(groupKey) : newExpanded.add(groupKey);
    setExpandedGroups(newExpanded);
  };

  const handleRowClick = (anlsIdx) => {
    navigate(`/noti-detail/${anlsIdx}`);
  };

  // 일간: 단순 리스트
  if (period === 'daily') {
    return (
      <div className="bordered-box grouped-detail-list">
        <h2 className="text-xl font-bold mb-4">상세 현황</h2>
        <div className="table-wrap scrl-custom">
          <table className="table border">
            <thead>
              <tr className="border-b">
                <th>탐지 시간</th>
                <th>탐지 구역</th>
                <th>해충 이름</th>
                <th>탐지 정확도</th>
              </tr>
            </thead>
            <tbody>
              {stats?.details?.length > 0 ? (
                stats.details.map((item, index) => (
                  <tr
                    key={index}
                    className="border-b hover:bg-gray-50 cursor-pointer"
                    onClick={() => handleRowClick(item.anlsIdx || index)}
                  >
                    <td>{item.time}</td>
                    <td>{item.greenhouse}</td>
                    <td>{item.insect}</td>
                    <td>{item.accuracy}%</td>
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

  // 월간/연간: 주차 또는 월별로 그룹핑
  const groupedData =
    period === 'monthly'
      ? groupDataByWeek(stats.details)
      : period === 'yearly'
      ? groupDataByMonth(stats.details)
      : null;

  return (
    <div className="bordered-box grouped-detail-list">
      <h2 className="text-xl font-bold mb-4">상세 현황</h2>

      {groupedData ? (
        <div className="space-y-2">
          {Object.entries(groupedData).map(([groupKey, group]) => (
            <div key={groupKey} className="border rounded-lg overflow-hidden">
              {/* 토글 헤더 */}
              <button
                className="w-full p-4 text-left hover:bg-gray-50 flex items-center justify-between"
                onClick={() => toggleGroup(groupKey)}
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">{expandedGroups.has(groupKey) ? '▼' : '▶'}</span>
                  <span className="font-semibold">{group.title}</span>
                  <span className="text-sm text-gray-600">({group.count}건)</span>
                </div>
              </button>

              {/* 토글 내용 */}
              {expandedGroups.has(groupKey) && (
                <div className="table-wrap scrl-custom">
                  <table className="table border">
                    <thead>
                      <tr className="bg-gray-50">
                        <th>탐지 시간</th>
                        <th>탐지 구역</th>
                        <th>해충 이름</th>
                        <th>탐지 정확도</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.items.map((item) => (
                        <tr
                          key={item.anlsIdx}
                          className="border-b hover:bg-blue-50 cursor-pointer"
                          onClick={() => handleRowClick(item.anlsIdx)}
                        >
                          <td>{item.datetime}</td>
                          <td>{item.region}</td>
                          <td>{item.bugType}</td>
                          <td>{item.accuracy}%</td>
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
        <div className="p-4 text-center text-gray-500">데이터가 없습니다.</div>
      )}
    </div>
  );
}
