// src/components/Home.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Home() {
  return(
    <div className="p-4">
      <p>
      해충 때문에 고민이신가요?
      든든하고 귀여운 백구가 밭을 지켜드립니다!
      카메라 <strong>실시간 감지</strong>로 해충의 움직임을 10초 만에 탐지해서, 9초 안에 알려 드려요
      </p>     
    </div>
  )

}
