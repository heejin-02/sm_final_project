// src/components/Home.jsx
// 초기화면
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import LoginForm from '../components/LoginForm'; // 로그인폼은 LoginForm 컴포넌트로 따로 분리

export default function Home({ onLogin }) {
  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')]">
      <div className="cont-wrap">
        <h2 className="font-semibold text-2xl">당신이 쉴 때도, 우리는 해충을 감지중입니다.</h2>
        <LoginForm onLogin={onLogin}/>
        <div className="text-xl">
        해충 때문에 고민이신가요? <br/>
        든든하고 귀여운 백구가 밭을 지켜드립니다! <br/>
        <span className='font-semibold'>카메라 실시간 감지</span>로 해충의 움직임을  <br/>
        <span className="font-semibold text-[var(--color-accent)]">10초</span> 만에 탐지해서, <span className="font-semibold text-[var(--color-accent)]">9초</span> 안에 알려 드려요
        </div> 
      </div>
    </div>
  )

}
