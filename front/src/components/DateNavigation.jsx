// src/components/DateNavigation.jsx
import React, { useState, useEffect } from 'react';

export default function DateNavigation({ period, currentDate, onDateChange }) {
  const [year, setYear] = useState(new Date(currentDate).getFullYear());
  const [month, setMonth] = useState(new Date(currentDate).getMonth() + 1);
  const [day, setDay] = useState(new Date(currentDate).getDate());

  // currentDate가 변경되면 상태 업데이트
  useEffect(() => {
    const date = new Date(currentDate);
    setYear(date.getFullYear());
    setMonth(date.getMonth() + 1);
    setDay(date.getDate());
  }, [currentDate]);

  // 오늘 날짜 확인
  const isDateToday = (date) => {
    const today = new Date();
    const checkDate = new Date(date);
    return today.toDateString() === checkDate.toDateString();
  };

  // 미래 날짜인지 확인
  const isFutureDate = (date) => {
    const today = new Date();
    const checkDate = new Date(date);
    today.setHours(0, 0, 0, 0);
    checkDate.setHours(0, 0, 0, 0);
    return checkDate > today;
  };

  // “다음” 버튼 활성 여부 (daily/monthly/yearly 별 로직 통합)
  const isNextDisabled = (() => {
    const today = new Date();
    const sel = new Date(currentDate);

    if (period === 'daily') {
      const next = new Date(sel);
      next.setDate(sel.getDate() + 1);
      return isFutureDate(next);
    }
    if (period === 'monthly') {
      return sel.getFullYear() === today.getFullYear()
          && sel.getMonth()    === today.getMonth();
    }
    if (period === 'yearly') {
      return sel.getFullYear() === today.getFullYear();
    }
    return false;
  })();  

  // 이전/다음 날짜로 이동
  const handlePrevious = () => {
    const currentDateObj = new Date(currentDate);
    let newDate;

    switch (period) {
      case 'daily':
        newDate = new Date(currentDateObj);
        newDate.setDate(currentDateObj.getDate() - 1);
        break;
      case 'monthly':
        newDate = new Date(currentDateObj);
        newDate.setMonth(currentDateObj.getMonth() - 1);
        break;
      case 'yearly':
        newDate = new Date(currentDateObj);
        newDate.setFullYear(currentDateObj.getFullYear() - 1);
        break;
      default:
        return;
    }
    onDateChange(newDate);
  };

  const handleNext = () => {
    const currentDateObj = new Date(currentDate);
    let newDate;

    switch (period) {
      case 'daily':
        newDate = new Date(currentDateObj);
        newDate.setDate(currentDateObj.getDate() + 1);
        // daily 모드에서는 오늘 이후로 이동 불가
        if (isFutureDate(newDate)) {
          return;
        }
        break;
      case 'monthly':
        newDate = new Date(currentDateObj);
        newDate.setMonth(currentDateObj.getMonth() + 1);
        break;
      case 'yearly':
        newDate = new Date(currentDateObj);
        newDate.setFullYear(currentDateObj.getFullYear() + 1);
        break;
      default:
        return;
    }
    onDateChange(newDate);
  };

  // 날짜 변경 버튼 클릭
  const handleDateChange = () => {
    let newDate;

    switch (period) {
      case 'daily':
        newDate = new Date(year, month - 1, day);
        break;
      case 'monthly':
        newDate = new Date(year, month - 1, 1);
        break;
      case 'yearly':
        newDate = new Date(year, 0, 1);
        break;
      default:
        return;
    }

    onDateChange(newDate);
  };

  // 연도 옵션 생성 (현재 연도 기준 ±5년)
  const getYearOptions = () => {
    const currentYear = new Date().getFullYear();
    const years = [];
    for (let i = currentYear - 5; i <= currentYear + 1; i++) {
      years.push(i);
    }
    return years;
  };

  // 월 옵션 생성
  const getMonthOptions = () => {
    return Array.from({ length: 12 }, (_, i) => i + 1);
  };

  // 일 옵션 생성 (해당 월의 마지막 날까지)
  const getDayOptions = () => {
    const daysInMonth = new Date(year, month, 0).getDate();
    return Array.from({ length: daysInMonth }, (_, i) => i + 1);
  };

  return (
    <div className="date-navigation">
      <div className="date-controls">
        {/* 이전 버튼 */}
        <button
          className="date-nav-btn"
          onClick={handlePrevious}
          title={`이전 ${period === 'daily' ? '일' : period === 'monthly' ? '월' : '연도'}`}
        >
          ◀ <span>이전 {period === 'daily' ? '일' : period === 'monthly' ? '월' : '연도'}</span>
        </button>

        {/* 연도 선택 */}
        <select
          value={year}
          onChange={(e) => setYear(parseInt(e.target.value))}
          className="date-select"
        >
          {getYearOptions().map(y => (
            <option key={y} value={y}>{y}년</option>
          ))}
        </select>

        {/* 월 선택 (연간이 아닌 경우) */}
        {period !== 'yearly' && (
          <select
            value={month}
            onChange={(e) => setMonth(parseInt(e.target.value))}
            className="date-select"
          >
            {getMonthOptions().map(m => (
              <option key={m} value={m}>{m}월</option>
            ))}
          </select>
        )}

        {/* 일 선택 (일간인 경우) */}
        {period === 'daily' && (
          <select
            value={day}
            onChange={(e) => setDay(parseInt(e.target.value))}
            className="date-select"
          >
            {getDayOptions().map(d => (
              <option key={d} value={d}>{d}일</option>
            ))}
          </select>
        )}

        {/* 날짜 변경 버튼 */}
        <button
          className="btn btn-accent date-change-btn"
          onClick={handleDateChange}
        >
          날짜변경
        </button>

        {/* 다음 버튼 */}
        <button
          className={`date-nav-btn ${isNextDisabled ? 'disabled' : ''}`}
          style={{ visibility: isNextDisabled ? 'hidden' : 'visible' }}
          onClick={handleNext}
          disabled={isNextDisabled}
          title={`다음 ${period === 'daily' ? '일' : period === 'monthly' ? '월' : '연도'}`}
        >
          <span>다음 {period === 'daily' ? '일' : period === 'monthly' ? '월' : '연도'}</span> ▶
        </button>
      </div>
    </div>
  );
}
