// src/contexts/AuthContext.js
import { createContext, useContext, useState, useEffect } from 'react';
import { loginCheck, checkSession, logout as apiLogout, getUserFarms } from '../api/auth';
import axios from 'axios';

// checkSession은 auth.js에서 import하여 사용

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

  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // 초기화 시 서버 세션에서 로그인 상태 확인
  useEffect(() => {
    const initializeAuth = async () => {

      try {
        const sessionData = await checkSession();

        if (sessionData.isAuthenticated && sessionData.user) {

          // 농장 리스트는 항상 최신 정보로 조회
          let selectedFarm = null;
          try {
            const farmListResponse = await getUserFarms(sessionData.user.userPhone);
            const farmList = farmListResponse.data;

            // 세션에 저장된 농장 ID로 선택된 농장 찾기
            const selectedFarmIdx = sessionData.user.selectedFarmIdx;
            selectedFarm = farmList.find(farm => farm.farmIdx === selectedFarmIdx) || farmList[0] || null;

          } catch (error) {
            console.error('농장 리스트 조회 실패:', error);
          }

          setUser({
            userName: sessionData.user.userName,
            userPhone: sessionData.user.userPhone,
            role: sessionData.user.role,
            selectedFarm: selectedFarm,
          });
          setIsLoggedIn(true);

          // localStorage에도 저장
          localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify({
            ...sessionData.user,
            selectedFarm: selectedFarm
          }));
          localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
          localStorage.setItem(STORAGE_KEYS.LOGIN_TIME, new Date().toISOString());
        } else {
          // 서버 세션이 없는 경우 - localStorage 확인
          const savedUser = localStorage.getItem(STORAGE_KEYS.USER);
          const savedLoginStatus = localStorage.getItem(STORAGE_KEYS.IS_LOGGED_IN);

          if (savedUser && savedLoginStatus === 'true') {
            const userData = JSON.parse(savedUser);
            setUser(userData);
            setIsLoggedIn(true);
          } else {
            console.log('🧹 localStorage 정리');
            localStorage.removeItem(STORAGE_KEYS.USER);
            localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
            localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
            setUser(null);
            setIsLoggedIn(false);
          }
        }
      } catch (error) {
        // 오류 시 localStorage 기반으로 fallback
        const savedUser = localStorage.getItem(STORAGE_KEYS.USER);
        const savedLoginStatus = localStorage.getItem(STORAGE_KEYS.IS_LOGGED_IN);

        if (savedUser && savedLoginStatus === 'true') {
          const userData = JSON.parse(savedUser);
          setUser(userData);
          setIsLoggedIn(true);
        } else {
          setUser(null);
          setIsLoggedIn(false);
        }
      }
    };

    initializeAuth();
  }, []);

  // 로그인 상태가 변경될 때마다 localStorage 업데이트
  useEffect(() => {
    if (isLoggedIn && user) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
      localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
      localStorage.setItem(STORAGE_KEYS.LOGIN_TIME, new Date().toISOString());
      // console.log('로그인 상태 저장됨');
    } else {
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
      localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
      // console.log('로그인 상태 삭제됨');
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

  const logout = async () => {
    try {
      // auth.js의 logout 함수 사용 (POST 방식)
      await apiLogout();

      // 로그아웃 시 초기화
      setUser(null);
      setIsLoggedIn(false);

      // localStorage 정리
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
      localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);

    } catch (error) {
      // 서버 로그아웃 실패해도 클라이언트는 로그아웃 처리
      console.error('서버 로그아웃 실패:', error);
      setUser(null);
      setIsLoggedIn(false);
      localStorage.clear();
    }
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