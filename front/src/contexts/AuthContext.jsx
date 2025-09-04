// src/contexts/AuthContext.js
import { createContext, useContext, useState, useEffect } from 'react';
import {
  loginCheck,
  checkSession,
  logout as apiLogout,
  getUserFarms,
} from '../api/auth';

// 1. Context 생성
// - 전역에서 로그인 상태를 공유하기 위해 AuthContext 사용
const AuthContext = createContext();

// localStorage 키 상수
// - 브라우저 저장소에 로그인 정보를 남겨 새로고침해도 유지되도록 함
const STORAGE_KEYS = {
  USER: 'auth_user', // 사용자 정보 (JSON 문자열)
  IS_LOGGED_IN: 'auth_is_logged_in', // 로그인 여부 (true/false)
  LOGIN_TIME: 'auth_login_time', // 마지막 로그인 시각 (ISO 문자열)
};

// 로그인 만료 시간 (7일)
// - 7일이 지나면 자동 로그아웃
// - 7일 이내에 다시 로그인하면 로그인 시간이 갱신되어 연장됨
const LOGIN_EXPIRY_DAYS = 7;

export function AuthProvider({ children }) {
  // user 상태: 현재 로그인한 사용자 정보
  const [user, setUser] = useState(() => {
    try {
      const savedUser = localStorage.getItem(STORAGE_KEYS.USER);
      return savedUser ? JSON.parse(savedUser) : null;
    } catch (error) {
      console.error('사용자 정보 복원 실패:', error);
      return null;
    }
  });

  // isLoggedIn 상태: 로그인 여부
  const [isLoggedIn, setIsLoggedIn] = useState(
    () => localStorage.getItem(STORAGE_KEYS.IS_LOGGED_IN) === 'true'
  );

  // -------------------------------
  // 초기화 시: 서버 세션 확인 + localStorage 만료 검사
  // -------------------------------
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const sessionData = await checkSession();

        if (sessionData.isAuthenticated && sessionData.user) {
          // ✅ 서버 세션이 유효할 때: 최신 사용자 정보 반영
          let selectedFarm = null;
          try {
            const farmListResponse = await getUserFarms(
              sessionData.user.userPhone
            );
            const farmList = farmListResponse.data;
            const selectedFarmIdx = sessionData.user.selectedFarmIdx;
            selectedFarm =
              farmList.find((farm) => farm.farmIdx === selectedFarmIdx) ||
              farmList[0] ||
              null;
          } catch (error) {
            console.error('농장 리스트 조회 실패:', error);
          }

          const newUser = {
            userName: sessionData.user.userName,
            userPhone: sessionData.user.userPhone,
            role: sessionData.user.role,
            selectedFarm,
          };

          setUser(newUser);
          setIsLoggedIn(true);

          // localStorage 갱신
          localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(newUser));
          localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
          localStorage.setItem(
            STORAGE_KEYS.LOGIN_TIME,
            new Date().toISOString()
          );
        } else {
          // ❌ 서버 세션이 없을 때: localStorage 기반 fallback
          validateLocalStorageLogin();
        }
      } catch (error) {
        console.error('세션 체크 실패:', error);
        // ❌ 서버 연결 실패: localStorage 기반 fallback
        validateLocalStorageLogin();
      }
    };

    initializeAuth();
  }, []);

  // -------------------------------
  // localStorage 로그인 상태 검증
  // -------------------------------
  const validateLocalStorageLogin = () => {
    const savedUser = localStorage.getItem(STORAGE_KEYS.USER);
    const savedLoginStatus = localStorage.getItem(STORAGE_KEYS.IS_LOGGED_IN);
    const savedLoginTime = localStorage.getItem(STORAGE_KEYS.LOGIN_TIME);

    if (savedUser && savedLoginStatus === 'true' && savedLoginTime) {
      const lastLoginDate = new Date(savedLoginTime);
      const now = new Date();
      const diffDays = (now - lastLoginDate) / (1000 * 60 * 60 * 24);

      if (diffDays > LOGIN_EXPIRY_DAYS) {
        // 7일이 지나면 자동 로그아웃
        console.log('로그인 만료: 마지막 로그인 이후 7일 경과');
        setUser(null);
        setIsLoggedIn(false);
        localStorage.removeItem(STORAGE_KEYS.USER);
        localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
        localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
      } else {
        // 아직 기간 안 지났으면 로그인 유지
        setUser(JSON.parse(savedUser));
        setIsLoggedIn(true);
      }
    } else {
      // localStorage에 유효한 로그인 정보가 없음 → 로그아웃 처리
      setUser(null);
      setIsLoggedIn(false);
    }
  };

  // -------------------------------
  // 로그인/로그아웃 시 localStorage 동기화
  // -------------------------------
  useEffect(() => {
    if (isLoggedIn && user) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
      localStorage.setItem(STORAGE_KEYS.IS_LOGGED_IN, 'true');
      localStorage.setItem(STORAGE_KEYS.LOGIN_TIME, new Date().toISOString());
    } else {
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
      localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
    }
  }, [isLoggedIn, user]);

  // -------------------------------
  // 로그인 함수
  // -------------------------------
  const login = async (id, pw) => {
    try {
      const response = await loginCheck(id, pw);

      const newUser = {
        userName: response.data.userName,
        userPhone: response.data.userPhone,
        role: response.data.role,
        joinedAt: response.data.joinedAt,
        selectedFarm: null,
      };

      setUser(newUser);
      setIsLoggedIn(true);

      // 로그인 시간 갱신 (7일 기준점)
      localStorage.setItem(STORAGE_KEYS.LOGIN_TIME, new Date().toISOString());

      return response;
    } catch (error) {
      throw error;
    }
  };

  // -------------------------------
  // 로그아웃 함수
  // -------------------------------
  const logout = async () => {
    try {
      await apiLogout(); // 서버 로그아웃 시도
    } catch (error) {
      console.error('서버 로그아웃 실패:', error);
    } finally {
      // 서버 실패 여부와 관계없이 클라이언트는 무조건 로그아웃 처리
      setUser(null);
      setIsLoggedIn(false);
      localStorage.removeItem(STORAGE_KEYS.USER);
      localStorage.removeItem(STORAGE_KEYS.IS_LOGGED_IN);
      localStorage.removeItem(STORAGE_KEYS.LOGIN_TIME);
    }
  };

  // -------------------------------
  // 농장 선택 함수
  // -------------------------------
  const selectFarm = (farm) => {
    setUser((prev) => {
      const updatedUser = { ...prev, selectedFarm: farm };
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(updatedUser));
      return updatedUser;
    });
  };

  // -------------------------------
  // Provider 반환
  // -------------------------------
  return (
    <AuthContext.Provider
      value={{ user, isLoggedIn, login, logout, selectFarm }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// 3. 커스텀 훅
export function useAuth() {
  return useContext(AuthContext);
}
