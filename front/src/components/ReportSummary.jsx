// components/ReportSummary.jsx
import React from 'react';
import { speak } from '../utils/speech';

//
// 숫자 변환 유틸
//
const nativeMap = {
  1: '한',
  2: '두',
  3: '세',
  4: '네',
  5: '다섯',
  6: '여섯',
  7: '일곱',
  8: '여덟',
  9: '아홉',
  10: '열',
};

// 한자어 수 변환 (백/천 단위까지 지원)
function numberToSino(num) {
  const digits = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구'];
  const small = ['', '십', '백', '천'];

  if (num === 0) return '영';
  let str = '';
  let i = 0;
  while (num > 0) {
    const digit = num % 10;
    if (digit > 0) str = digits[digit] + small[i] + str;
    num = Math.floor(num / 10);
    i++;
  }
  return str;
}

// 단위별 규칙 적용
function numberToKorean(num, unit) {
  if ((unit === '마리' || unit === '건') && num >= 1 && num <= 10) {
    return `${nativeMap[num]} ${unit}`;
  }
  return `${numberToSino(num)} ${unit}`;
}

// 시간 전용 (항상 한자어)
function numberToHour(num) {
  return `${numberToSino(num)} 시`;
}

//
// 날짜 텍스트 변환
//
function formatDateText(date, period) {
  const today = new Date();
  const y = date.getFullYear();
  const m = date.getMonth() + 1;
  const d = date.getDate();

  const ty = today.getFullYear();
  const tm = today.getMonth() + 1;
  const td = today.getDate();

  if (period === 'daily') {
    if (y === ty && m === tm && d === td) return '오늘';
    const yesterday = new Date(today);
    yesterday.setDate(td - 1);
    if (
      y === yesterday.getFullYear() &&
      m === yesterday.getMonth() + 1 &&
      d === yesterday.getDate()
    ) {
      return '어제';
    }
    return `${y}년 ${m}월 ${d}일`;
  }

  if (period === 'monthly') {
    if (y === ty && m === tm) return '이번 달';
    const lastMonth = new Date(today);
    lastMonth.setMonth(tm - 1);
    if (y === lastMonth.getFullYear() && m === lastMonth.getMonth() + 1) {
      return '저번 달';
    }
    return `${y}년 ${m}월`;
  }

  if (period === 'yearly') {
    if (y === ty) return '올해';
    if (y === ty - 1) return '작년';
    return `${y}년`;
  }

  return '';
}

//
// ReportSummary 컴포넌트
//
export default function ReportSummary({
  stats,
  period,
  currentDate,
  gptSummary,
}) {
  const handleReadSummary = () => {
    const dateText = formatDateText(currentDate, period);

    // 총 탐지 수 & 해충 종류 수
    const total = stats?.totalCount ?? 0;
    const types = stats?.insectTypeCount ?? 0;
    const totalText = numberToKorean(total, '마리');
    const typeText = numberToSino(types) + ' 종';

    // 최다 탐지 구역 (동률 처리)
    let zoneText = '데이터 없음';
    if (Array.isArray(stats?.zoneStats) && stats.zoneStats.length > 0) {
      const maxCount = Math.max(...stats.zoneStats.map((z) => z.count));
      const topZones = stats.zoneStats.filter((z) => z.count === maxCount);

      const zoneNames = topZones.map(
        (z) =>
          z.zone.replace(
            /(\d+)번/,
            (_, num) => `${numberToSino(parseInt(num))} 번`
          ) + ` (${numberToKorean(z.count, '건')})`
      );

      zoneText = zoneNames.join(', ');
    }

    // 시간별 최댓값 요약
    let timeSummary = '';
    if (Array.isArray(stats?.hourlyStats) && stats.hourlyStats.length > 0) {
      const maxHourData = stats.hourlyStats.reduce(
        (max, item) => (item.count > max.count ? item : max),
        { hour: null, count: 0 }
      );
      if (maxHourData.count > 0) {
        timeSummary = `${numberToHour(
          parseInt(maxHourData.hour)
        )}에 ${numberToKorean(
          maxHourData.count,
          '마리'
        )}로 가장 많이 탐지되었습니다.`;
      }
    }

    // 해충 분포 요약 (최다 해충)
    let insectSummary = '';
    if (
      Array.isArray(stats?.insectDistribution) &&
      stats.insectDistribution.length > 0
    ) {
      const maxCount = Math.max(
        ...stats.insectDistribution.map((i) => i.count)
      );
      const topInsects = stats.insectDistribution.filter(
        (i) => i.count === maxCount
      );

      const insectNames = topInsects.map(
        (i) => `${i.insect} ${numberToKorean(i.count, '마리')}`
      );
      insectSummary = `가장 많이 탐지된 해충은 ${insectNames.join(
        ', '
      )}입니다.`;
    }

    // 최종 텍스트
    const text = `
      ${gptSummary ? '백구의 요약: ' + gptSummary + '.' : ''}
      ${dateText} 기준, 총 ${totalText}의 해충이 탐지되었습니다.
      탐지된 해충 종류는 ${typeText}이며,
      최다 탐지 구역은 ${zoneText}입니다.
      ${timeSummary}
      ${insectSummary}
    `;

    speak(text);
  };

  return (
    <div>
      <button
        onClick={handleReadSummary}
        className='mt-4 px-6 py-3 text-lg bg-green-600 text-white rounded-xl w-full'
      >
        🎙️ 전체 통계 음성 요약 듣기
      </button>
    </div>
  );
}
