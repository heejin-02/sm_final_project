import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

const WeatherContext = createContext();

const API_KEY = '019d414c565826322ad2f0b73af0129b';
const WEATHER_CACHE_DURATION = 10 * 60 * 1000; // 10분
const weatherCache = new Map();

const weatherKoreanMap = {
  Clear: '맑음',
  Clouds: '흐림',
  Rain: '비',
  Drizzle: '이슬비',
  Thunderstorm: '뇌우',
  Snow: '눈',
  Mist: '옅은 안개',
  Smoke: '연기',
  Haze: '실안개',
  Dust: '먼지',
  Fog: '짙은 안개',
  Sand: '황사',
  Ash: '화산재',
  Squall: '돌풍',
  Tornado: '토네이도',
};

const getWeatherIcon = (iconCode) =>
  `https://openweathermap.org/img/wn/${iconCode}@2x.png`;

// 주소 → 좌표
async function getCoordinatesFromAddress(address) {
  try {
    const url = `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(
      address
    )}&limit=1&appid=${API_KEY}`;
    const res = await axios.get(url);
    if (res.data?.length > 0) {
      const { lat, lon, name } = res.data[0];
      return { lat, lon, locationName: name };
    }
    return null;
  } catch {
    return null;
  }
}

// 현재 위치
async function getCurrentLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation 미지원'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        resolve({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          locationName: '현재 위치',
        });
      },
      (err) => reject(err),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
    );
  });
}

export function WeatherProvider({ children }) {
  const { user } = useAuth();
  const [weather, setWeather] = useState(() => {
    try {
      const saved = localStorage.getItem('lastWeatherData');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(!weather);
  const [lastFetchedAt, setLastFetchedAt] = useState(null);

  const fetchWeather = useCallback(async () => {
    const farmAddr = user?.selectedFarm?.farmAddr;
    const cacheKey = farmAddr || 'default_location';

    // 캐시 확인
    const cached = weatherCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < WEATHER_CACHE_DURATION) {
      setWeather(cached.data);
      setLoading(false);
      setLastFetchedAt(cached.timestamp);
      return cached.data;
    }

    setLoading(!weather); // 기존 데이터 있으면 false 유지
    try {
      let lat, lon, locationName;

      if (farmAddr) {
        const coords = await getCoordinatesFromAddress(farmAddr);
        if (coords) {
          lat = coords.lat;
          lon = coords.lon;
          locationName = coords.locationName;
        }
      }

      // 농장 좌표 없으면 현재 위치 fallback
      if (!lat || !lon) {
        try {
          const loc = await getCurrentLocation();
          lat = loc.lat;
          lon = loc.lon;
          locationName = loc.locationName;
        } catch {
          // 현재 위치 실패 시 서울 fallback
          lat = 37.5665;
          lon = 126.978;
          locationName = '서울';
        }
      }

      // API 호출
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`;
      const res = await axios.get(url);
      const data = res.data;

      const englishCondition = data.weather[0].main;
      const koreanCondition =
        weatherKoreanMap[englishCondition] || englishCondition;

      const weatherData = {
        cityName: data.name,
        cityKorean: farmAddr || locationName,
        farmAddr: farmAddr || locationName,
        temp: data.main.temp,
        tempMin: data.main.temp_min,
        tempMax: data.main.temp_max,
        humidity: data.main.humidity,
        wind: data.wind.speed,
        condition: koreanCondition,
        iconUrl: getWeatherIcon(data.weather[0].icon),
      };

      weatherCache.set(cacheKey, { data: weatherData, timestamp: Date.now() });
      localStorage.setItem('lastWeatherData', JSON.stringify(weatherData));
      setWeather(weatherData);
      setLastFetchedAt(Date.now());
      setLoading(false);
      return weatherData;
    } catch (err) {
      console.error('날씨 API 실패:', err);
      setLoading(false);
      return null;
    }
  }, [user?.selectedFarm?.farmAddr, weather]);

  useEffect(() => {
    fetchWeather();
    const interval = setInterval(fetchWeather, WEATHER_CACHE_DURATION);
    return () => clearInterval(interval);
  }, [fetchWeather]);

  return (
    <WeatherContext.Provider
      value={{ weather, loading, fetchWeather, lastFetchedAt }}
    >
      {children}
    </WeatherContext.Provider>
  );
}

export const useWeather = () => useContext(WeatherContext);
