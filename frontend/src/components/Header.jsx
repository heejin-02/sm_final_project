import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
import logo from '/images/logo.svg';
import Weather from './Weather';

export default function Header() {
  const { user } = useAuth(); // 전역에서 로그인된 사용자 정보 가져옴
  const navigate = useNavigate();
  
  return (
    <header className="header flex align-center justify-between p-4 fixed">

      <div className="user-area">
        {user?.name && <p className="text-sm">{user.name} 님</p>}
        {/* 선택된 농장 이름이 있으면 */}
        {user?.selectedFarm?.name && (
          <p className="text-sm">
            현재 농장: <span className="font-semibold">{user.selectedFarm.name}</span>
          </p>
        )}
      </div>

      <div className="logo-area center-absolute cursor-pointer" onClick={() => navigate('/')}>
        <img src={logo} alt="로고" className="logo" />
      </div>

      <div className="">
        <Weather />
      </div>

    </header>
  );
}
