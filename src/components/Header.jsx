import React from "react";
import logo from '../assets/images/logo.png';
import Weather from './Weather';


function Header({ user }) {
  return (
    <header className="header flex align-center justify-between p-4">

      <div className="user-area">
        {user && <span>{user.name} 님 👋</span>}
      </div>

      <div className="logo-area">
        <img src={logo} alt="로고" className="logo" width="150" />
      </div>

      <div className="">
        <Weather />
      </div>

    </header>
  );
}

export default Header;
