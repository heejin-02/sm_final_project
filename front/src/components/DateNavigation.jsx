// src/components/DateNavigation.jsx
import React, { useState } from 'react';

export default function DateNavigation({ period, currentDate, onDateChange }) {
  const [showModal, setShowModal] = useState(false);

  // 현재 날짜를 보기 좋게 포맷
  const formatDate = (date, period) => {
    const d = new Date(date);
    switch (period) {
      case 'daily':
        return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일`;
      case 'monthly':
        return `${d.getFullYear()}년 ${d.getMonth() + 1}월`;
      case 'yearly':
        return `${d.getFullYear()}년`;
      default:
        return date;
    }
  };

  // 빠른 선택 옵션들
  const getQuickOptions = () => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    
    const thisWeekStart = new Date(today);
    thisWeekStart.setDate(today.getDate() - today.getDay());
    
    const lastWeekStart = new Date(thisWeekStart);
    lastWeekStart.setDate(thisWeekStart.getDate() - 7);
    
    const thisMonth = new Date(today.getFullYear(), today.getMonth(), 1);
    const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    const threeMonthsAgo = new Date(today.getFullYear(), today.getMonth() - 3, 1);
    const sixMonthsAgo = new Date(today.getFullYear(), today.getMonth() - 6, 1);
    const lastYearSameMonth = new Date(today.getFullYear() - 1, today.getMonth(), 1);
    
    const thisYear = new Date(today.getFullYear(), 0, 1);
    const lastYear = new Date(today.getFullYear() - 1, 0, 1);
    const twoYearsAgo = new Date(today.getFullYear() - 2, 0, 1);

    switch (period) {
      case 'daily':
        return [
          { label: '오늘', date: today },
          { label: '어제', date: yesterday },
          { label: '이번주', date: thisWeekStart },
          { label: '지난주', date: lastWeekStart },
        ];
      case 'monthly':
        return [
          { label: '이번달', date: thisMonth },
          { label: '지난달', date: lastMonth },
          { label: '3개월전', date: threeMonthsAgo },
          { label: '6개월전', date: sixMonthsAgo },
          { label: '작년 동기', date: lastYearSameMonth },
        ];
      case 'yearly':
        return [
          { label: '올해', date: thisYear },
          { label: '작년', date: lastYear },
          { label: '2년전', date: twoYearsAgo },
        ];
      default:
        return [];
    }
  };

  const handleQuickSelect = (selectedDate) => {
    onDateChange(selectedDate);
    setShowModal(false);
  };

  const handleDirectSelect = () => {
    // TODO: 직접 선택 모달 구현
    alert('직접 선택 기능은 추후 구현 예정입니다.');
    setShowModal(false);
  };

  return (
    <div className="date-navigation">
      {/* 현재 날짜 표시 */}
      <div className="current-date-display">
        <div className="current-date-text">
          {formatDate(currentDate, period)}
        </div>
        <button 
          className="date-change-btn"
          onClick={() => setShowModal(true)}
        >
          날짜 변경
        </button>
      </div>

      {/* 빠른 선택 모달 */}
      {showModal && (
        <div className="date-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="date-modal" onClick={(e) => e.stopPropagation()}>
            <div className="date-modal-header">
              <h3>날짜 선택</h3>
              <button 
                className="modal-close-btn"
                onClick={() => setShowModal(false)}
              >
                ✕
              </button>
            </div>
            
            <div className="quick-options">
              {getQuickOptions().map((option, index) => (
                <button
                  key={index}
                  className="quick-option-btn"
                  onClick={() => handleQuickSelect(option.date)}
                >
                  {option.label}
                </button>
              ))}
              
              <button
                className="quick-option-btn direct-select"
                onClick={handleDirectSelect}
              >
                직접 선택
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
