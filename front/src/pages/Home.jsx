// src/components/Home.jsx
// 초기화면
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import LoginForm from '../components/LoginForm'; // 로그인폼은 LoginForm 컴포넌트로 따로 분리

export default function Home({ onLogin }) {
  return(
    <div className="section home flex flex-col bg-[url('/images/home_bg2.jpg')] bg-center bg-cover">
      
      <div className="cont-wrap">
       
        <div className="home-lt">
          <h2 className="home-tit">당신이 쉴 때도, <br/>우리는 해충을 감지합니다.</h2>   
          <div className="home-desc">
          해충 때문에 고민이신가요? <br/>
          든든하고 귀여운 백구가 밭을 지켜드립니다!
          </div>
          <button className="btn btn-primary btn-xl">상담신청</button>
        </div>

        <div className="white-box home-rt">
          <div className="desc color-80">
            <span className='text-black'>카메라 실시간 감지</span>를 통해 <br />
            해충의 움직임을 <span className='color-primary'>10초</span> 만에 탐지, <span className='color-primary'>9초</span> 안에 알려드립니다
          </div>
          <LoginForm onLogin={onLogin}/>
        </div> 
      
      </div>

    </div>
  )

}
