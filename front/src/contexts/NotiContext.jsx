// src/contexts/NotiContext.jsx
import { createContext, useContext, useState } from 'react';

const NotiContext = createContext();
export function NotiProvider({ children }) {
  const [isNotiOpen, setIsNotiOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [cachedAlerts, setCachedAlerts] = useState(() => {
    try {
      const saved = localStorage.getItem('cachedAlerts');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  return (
    <NotiContext.Provider
      value={{
        isNotiOpen,
        setIsNotiOpen,
        unreadCount,
        setUnreadCount,
        cachedAlerts,
        setCachedAlerts,
      }}
    >
      {children}
    </NotiContext.Provider>
  );
}

export function useNoti() {
  return useContext(NotiContext);
}
