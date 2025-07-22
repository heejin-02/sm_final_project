import React from "react";
import logo from '/images/logo.svg';
import Weather from './Weather';
import { useAuth } from '../contexts/AuthContext';

export default function Header({ }) {
  const { user } = useAuth(); // 전역에서 로그인된 사용자 정보 가져옴

  return (
    <header className="header flex align-center justify-between p-4 fixed">

      <div className="user-area">
        {user?.name && <p>{user.name} 님</p>}
        {user?.farmName && <p>{user.farmName} 관리중</p>}  
      </div>

      <div className="logo-area center-absolute">
        <img src={logo} alt="로고" className="logo" width="150" />
      </div>

      <div className="">
        <Weather />
      </div>

    </header>
  );
}
