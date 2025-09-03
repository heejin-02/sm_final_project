// src/contexts/NotiContext.jsx
import { createContext, useContext, useState } from 'react';

const NotiContext = createContext();

export function NotiProvider({ children }) {
    const [isNotiOpen, setIsNotiOpen] = useState(false);
    const [unreadCount, setUnreadCount] = useState(0); // 추가

    return <NotiContext.Provider value={{ isNotiOpen, setIsNotiOpen, unreadCount, setUnreadCount }}>{children}</NotiContext.Provider>;
}

export function useNoti() {
    return useContext(NotiContext);
}
