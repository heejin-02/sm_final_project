// contexts/NotiContext.jsx
import { createContext, useContext, useState } from 'react';

const NotiContext = createContext();

export function NotiProvider({ children }) {
    const [isNotiOpen, setIsNotiOpen] = useState(false);

    return <NotiContext.Provider value={{ isNotiOpen, setIsNotiOpen }}>{children}</NotiContext.Provider>;
}

export function useNoti() {
    return useContext(NotiContext);
}
