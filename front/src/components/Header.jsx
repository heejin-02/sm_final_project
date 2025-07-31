import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from '../contexts/AuthContext';
import logo from '/images/logo-horizon.svg';
import Weather from './Weather';
import { LuLogOut } from "react-icons/lu";

export default function Header() {
  const { user, logout } = useAuth(); // logout 함수도 가져옴
  const navigate = useNavigate();

  const handleLogout = () => {
    logout(); // AuthContext의 logout 함수 호출
    navigate('/', { replace: true }); // home.jsx로 이동
  };
  
  return (
    <header className="header">
      <div className="logo-area cursor-pointer" onClick={() => navigate('/')}>
        <img src={logo} alt="로고" className="logo" />
      </div>

      <div className="user-area">
        {user?.userName && user.role !== 'admin' && (
          (() => {
            return !user.selectedFarm?.farmName ? (
              <span className="text-2xl">{user.userName}님 환영합니다</span>
            ) : (
              <span className="text-2xl">
                {user.userName}님의&nbsp;
                <span className="font-semibold">{user.selectedFarm.farmName}</span>
              </span>
            );
          })()
        )}
      </div>

      <div className="wheather-box">
        {user?.role !== 'admin' ? (
          <>
            <Weather />
            {/* <button className="btn btn-sm" onClick={handleLogout}>로그아웃</button> */}
          </>
        ) : (
          <div className="flex items-center gap-2">
            <span className="">관리자 모드</span>
            <LuLogOut size={24} onClick={handleLogout} className="cursor-pointer"/>
          </div>
        )}
      </div>
    </header>
  );
}
