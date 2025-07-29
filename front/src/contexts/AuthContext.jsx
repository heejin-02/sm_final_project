// src/contexts/AuthContext.js
import React, { createContext, useContext, useState } from 'react';
import { loginCheck } from '../api/auth';

// 1. Context 생성
const AuthContext = createContext();

// 2. Provider 컴포넌트 정의
export function AuthProvider({ children }) {

  // 로그인한 사용자 정보
  const [user, setUser] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // 로그인 + 로그인한 유저 정보 저장
  const login = async (id, pw) => {
    try {
      const response = await loginCheck(id, pw);

      // user에 모든 사용자 정보 저장
      setUser({
        userName: response.data.userName,
        userPhone: response.data.userPhone,
        role: response.data.role,
        joinedAt: response.data.joinedAt,
        selectedFarm: null,
      });

      setIsLoggedIn(true);
      return response;
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    // 로그아웃 시 초기화
    setUser(null);
    setIsLoggedIn(false);
  };

  const selectFarm = (farm) => {
    // 농장정보 받아오기
    setUser((prev) => ({ ...prev, selectedFarm: farm }));
  };

  return (
    <AuthContext.Provider value={{ user, isLoggedIn, login, logout, selectFarm }}>
      {children}
    </AuthContext.Provider>
  );
}

// 3. 커스텀 훅으로 context 사용
export function useAuth() {
  return useContext(AuthContext);
}