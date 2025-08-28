// src/contexts/DataCacheContext.jsx
import { createContext, useContext, useState, useEffect } from 'react';

const DataCacheContext = createContext();

// 캐시 만료 시간 (30분)
const CACHE_DURATION = 30 * 60 * 1000;

export function DataCacheProvider({ children }) {
  // 전역 캐시 상태
  const [cache, setCache] = useState(() => {
    // localStorage에서 캐시 복원
    try {
      const saved = localStorage.getItem('farmDataCache');
      if (saved) {
        const parsed = JSON.parse(saved);
        // 만료된 캐시 제거
        const now = Date.now();
        const validCache = {};
        Object.keys(parsed).forEach(key => {
          if (parsed[key].timestamp && (now - parsed[key].timestamp) < CACHE_DURATION) {
            validCache[key] = parsed[key];
          }
        });
        return validCache;
      }
    } catch (error) {
      // console.log('캐시 복원 실패');
    }
    return {};
  });

  // 캐시 저장 (localStorage 동기화)
  useEffect(() => {
    try {
      localStorage.setItem('farmDataCache', JSON.stringify(cache));
    } catch (error) {
      // console.log('캐시 저장 실패');
    }
  }, [cache]);

  // 캐시 설정
  const setData = (key, data) => {
    setCache(prev => ({
      ...prev,
      [key]: {
        data,
        timestamp: Date.now()
      }
    }));
  };

  // 캐시 조회
  const getData = (key) => {
    const cached = cache[key];
    if (!cached) return null;
    
    // 만료 체크
    if (Date.now() - cached.timestamp > CACHE_DURATION) {
      // 만료된 캐시 제거
      setCache(prev => {
        const newCache = { ...prev };
        delete newCache[key];
        return newCache;
      });
      return null;
    }
    
    return cached.data;
  };

  // 캐시 존재 여부 확인
  const hasData = (key) => {
    return getData(key) !== null;
  };

  // 캐시 삭제
  const clearData = (key) => {
    setCache(prev => {
      const newCache = { ...prev };
      delete newCache[key];
      return newCache;
    });
  };

  // 전체 캐시 삭제
  const clearAllData = () => {
    setCache({});
  };

  // 구역 데이터 전용 함수들
  const setGreenhouseData = (farmIdx, greenhouseData) => {
    const key = `greenhouse_data_${farmIdx}`;
    setData(key, greenhouseData);
  };

  const getGreenhouseData = (farmIdx) => {
    const key = `greenhouse_data_${farmIdx}`;
    return getData(key);
  };

  // ghName으로 ghIdx 찾기 (안전한 매칭)
  const findGhIdxByName = (farmIdx, ghName) => {
    const greenhouseData = getGreenhouseData(farmIdx);
    if (!greenhouseData || !ghName) return null;

    const found = greenhouseData.find(gh => gh.ghName === ghName);
    return found ? found.ghIdx : null;
  };

  const value = {
    setData,
    getData,
    hasData,
    clearData,
    clearAllData,
    // 구역 데이터 전용 함수들
    setGreenhouseData,
    getGreenhouseData,
    findGhIdxByName
  };

  return (
    <DataCacheContext.Provider value={value}>
      {children}
    </DataCacheContext.Provider>
  );
}

export function useDataCache() {
  const context = useContext(DataCacheContext);
  if (!context) {
    throw new Error('useDataCache must be used within a DataCacheProvider');
  }
  return context;
}
