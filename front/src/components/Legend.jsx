// 지도 시각화 색상 범례(gradient bar + 텍스트)
import React from 'react';

export default function Legend({ min, max }) {
  return (
    <div className='legend'>
      {/* 3컬러 그라데이션 바 */}
      <div className='gradient-bar' />

      {/* 퍼센트 눈금 */}
      <div className='labels'>
        <span className='text-xs font-semibold'>위험</span>
        <span className='text-xs font-semibold'>주의</span>
        <span className='text-xs font-semibold'>안심</span>
      </div>
    </div>
  );
}
