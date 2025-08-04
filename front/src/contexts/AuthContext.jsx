// src/contexts/AuthContext.js
import { createContext, useContext, useState, useEffect } from 'react';
import { loginCheck } from '../api/auth';

// 1. Context 생성
const AuthContext = createContext();

// localStorage 키 상수
const STORAGE_KEYS = {
  USER: 'auth_user',
  IS_LOGGED_IN: 'auth_is_logged_in',
  LOGIN_TIME: 'auth_login_time'
};

// 로그인 만료 시간 (7일)
const LOGIN_EXPIRY_DAYS = 7;

// 2. Provider 컴포넌트 정의
export function AuthProvider({ children }) {

  // 로그인한 사용자 정보 (localStorage에서 복원)
  const [user, setUser] = useState(() => {
    try {
      const savedUser = localStorage.getItem(STORAGE_KEYS.USER);
      return savedUser ? JSON.parse(savedUser) : null;
    } catch (error) {
      console.error('사용자 정보 복원 실패:', error);
      return null;
    }
  });

  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    try {
      const savedLoginState = localStorage.getItem(STORAGE_KEYS.IS_LOGGED_IN);
      const loginTime = localStorage.getItem(STORAGE_KEYS.LOGIN_TIME);

      if (savedLoginState === 'true' && loginTime) {
        // 로그인 시간 체크 (7일 이내인지)
        const loginDate = new Date(loginTime);
        const now = new Date();
        const daysDiff = (now - loginDate) / (1000 * 60 * 60 * 24);

        if (daysDiff <= LOGIN_EXPIRY_DAYS) {
          return true;
        } else {
          // 만료된 로그인 정보 삭제
          localStorage.removeItem(STORAGE_KEYS.USER);
          localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
          localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
          return false;
        }
      }
      return false;
    } catch (error) {
      console.error('로그인 상태 복원 실패:', error);
      return false;
    }
  });

  // 로그인 상태가 변경될 때마다 localStorage 업데이트
  useEffect(() => {
    if (isLoggedIn && user) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
      localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
      localStorage.setItem(STORAGE_KEYS.LOGIN_TIME, new Date().toISOString());
      console.log('✅ 로그인 상태 저장됨');
    } else {
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
      localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
      console.log('🗑️ 로그인 상태 삭제됨');
    }
  }, [isLoggedIn, user]);

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

    // localStorage 정리
    localStorage.removeItem(STORAGE_KEYS.USER);
    localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
    localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);

    console.log('👋 로그아웃 완료');
  };

  const selectFarm = (farm) => {
    // 농장정보 받아오기
    setUser((prev) => {
      const updatedUser = { ...prev, selectedFarm: farm };
      // localStorage에도 즉시 업데이트
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(updatedUser));
      return updatedUser;
    });
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