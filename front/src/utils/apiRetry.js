// src/utils/apiRetry.js
// API 재시도 유틸리티

export const retryApiCall = async (apiFunction, maxRetries = 2, delay = 1000) => {
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await apiFunction();
      return result;
    } catch (error) {
      lastError = error;
      
      // 마지막 시도가 아니면 재시도
      if (attempt < maxRetries) {
        console.log(`API 호출 실패 (${attempt + 1}/${maxRetries + 1}), ${delay}ms 후 재시도...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        delay *= 1.5; // 지수 백오프
      }
    }
  }
  
  throw lastError;
};

// 빠른 실패를 위한 타임아웃 래퍼
export const withTimeout = (promise, timeoutMs = 5000) => {
  return Promise.race([
    promise,
    new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Request timeout')), timeoutMs)
    )
  ]);
};

// 네트워크 상태 확인
export const isNetworkError = (error) => {
  return error.code === 'ERR_NETWORK' || 
         error.code === 'ERR_CONNECTION_TIMED_OUT' ||
         error.message?.includes('Network Error') ||
         error.message?.includes('timeout');
};
