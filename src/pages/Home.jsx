// src/components/Home.jsx
// 초기화면
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import LoginForm from '../components/LoginForm'; // 로그인폼은 LoginForm 컴포넌트로 따로 분리
import bgImg from '/images/bg_home.jpg';

export default function Home() {
  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center flex flex-col justify-center gap-6">
        <h2 className="font-bold text-2xl">당신이 쉴 때도, 우리는 해충을 감지중입니다.</h2>
        <LoginForm/>
        <div className="text-xl">
        해충 때문에 고민이신가요? <br/>
        든든하고 귀여운 백구가 밭을 지켜드립니다! <br/>
        카메라 <strong>실시간 감지</strong>로 해충의 움직임을  <br/>
        10초 만에 탐지해서, 9초 안에 알려 드려요
        </div> 
      </div>
    </div>
  )

}
