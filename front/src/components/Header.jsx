import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
import logo from '/images/logo-horizon.svg';
import Weather from './Weather';

export default function Header() {
  const { user } = useAuth(); // 전역에서 로그인된 사용자 정보 가져옴
  const navigate = useNavigate();
  
  return (
    <header className="header flex align-center justify-between p-4 fixed">
     
      <div className="logo-area cursor-pointer" onClick={() => navigate('/')}>
        <img src={logo} alt="로고" className="logo" />
      </div>

      <div className="user-area">
        {user?.name && (
          !user.selectedFarm?.name ? (
            <span className="text-2xl">{user.name}님 환영합니다</span>
          ) : (
            <span className="text-2xl">
              {user.name}님의&nbsp;
              <span className="font-semibold">{user.selectedFarm.name}</span>
            </span>
          )
        )}
      </div>

      <div className="">
        <Weather />
      </div>

    </header>
  );
}
