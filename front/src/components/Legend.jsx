// 지도 시각화 색상 범례(gradient bar + 텍스트)
import React from "react";

export default function Legend({ min, max }) {

  return (
    <div className="flex items-top gap-1 absolute top-1 left-1 bg-white p-1 rounded">
      {/* 3컬러 그라데이션 바 */}
      <div
        className="w-2 h-auto rounded bg-gradient-to-t from-green-500 via-yellow-300 to-red-500"
      />

      {/* 퍼센트 눈금 */}
      <div className="flex flex-col justify-between text-xs text-gray-600">
        <span className="text-xs font-semibold">위험</span>
        <span className="text-xs font-semibold">주의</span>
        <span className="text-xs font-semibold">안심</span>
      </div>
    </div>
  );
}
