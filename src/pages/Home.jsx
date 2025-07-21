// src/components/Home.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import LoginForm from '../components/LoginForm';

export default function Home() {
  return(
    <div className="p-4 text-center">
      <h2 className="font-bold text-2xl">당신이 쉴 때도, 우리는 해충을 감지중입니다.</h2>
      <LoginForm/>
      <div className="text-xl">
      해충 때문에 고민이신가요? <br/>
      든든하고 귀여운 백구가 밭을 지켜드립니다! <br/>
      카메라 <strong>실시간 감지</strong>로 해충의 움직임을  <br/>
      10초 만에 탐지해서, 9초 안에 알려 드려요
      </div>     
    </div>
  )

}
