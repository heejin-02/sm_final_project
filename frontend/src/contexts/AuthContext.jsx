// src/contexts/AuthContext.js
import { createContext, useContext, useState } from 'react';

// 1. Context 생성
const AuthContext = createContext();

// 2. Provider 컴포넌트 정의
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null); // 로그인한 사용자 정보

  const login = (userData) => setUser(userData); // 로그인 시 정보 저장
  const logout = () => setUser(null);            // 로그아웃 시 초기화

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// 3. 커스텀 훅으로 context 사용
export function useAuth() {
  return useContext(AuthContext);
}