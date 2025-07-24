// src/contexts/AuthContext.js
import React, { createContext, useContext, useState } from 'react';

// 1. Context 생성
const AuthContext = createContext();

// 2. Provider 컴포넌트 정의
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null); // 로그인한 사용자 정보

  const login = (userData) => {
    // 로그인 시 정보 저장
    setUser(userData); 
  };

  const logout = () => {
    // 로그아웃 시 초기화
    setUser(null) 
  };   
  
  const selectFarm = (farm) => { 
    // 농장정보 받아오기
    setUser((prev) => ({ ...prev, selectedFarm: farm }));
  };  

  return (
    <AuthContext.Provider value={{ user, login, logout, selectFarm }}>
      {children}
    </AuthContext.Provider>
  );
}

// 3. 커스텀 훅으로 context 사용
export function useAuth() {
  return useContext(AuthContext);
}