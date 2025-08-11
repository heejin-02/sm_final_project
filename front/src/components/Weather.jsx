// openweathermap API 사용

import { useEffect, useState } from 'react';
import axios from 'axios';
import Loader from './Loader';
import { useAuth } from '../contexts/AuthContext';


const API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;

// 날씨 데이터 캐시 (30분 유지)
const WEATHER_CACHE_DURATION = 30 * 60 * 1000; // 30분
const weatherCache = new Map();

const cityKoreanMap = {
  Seoul: '서울',
  Busan: '부산',
  Incheon: '인천',
  Daegu: '대구',
  Daejeon: '대전',
  Gwangju: '광주',
  Ulsan: '울산',
  Suwon: '수원',
  Changwon: '창원',
  Seongnam: '성남',
  Jeju: '제주',
};

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

// 날씨 아이콘 매핑 (OpenWeatherMap 공식 아이콘 사용)
const getWeatherIcon = (iconCode) => {
  // OpenWeatherMap 아이콘 URL
  return `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
};

// 커스텀 아이콘으로 바꿀 때 사용할 매핑 (나중에 활성화)
const customWeatherIcons = {
  Clear: '/images/weather/clear.png',
  Clouds: '/images/weather/clouds.png',
  Rain: '/images/weather/rain.png',
  Drizzle: '/images/weather/drizzle.png',
  Thunderstorm: '/images/weather/thunderstorm.png',
  Snow: '/images/weather/snow.png',
  Mist: '/images/weather/mist.png',
  Fog: '/images/weather/mist.png',
  Smoke: '/images/weather/mist.png',
  Haze: '/images/weather/mist.png',
  Dust: '/images/weather/dust.png',
  Sand: '/images/weather/dust.png',
  Ash: '/images/weather/dust.png',
  Squall: '/images/weather/wind.png',
  Tornado: '/images/weather/wind.png',
};

// 주소를 좌표로 변환하는 함수
const getCoordinatesFromAddress = async (address) => {
  try {
    const geocodeUrl = `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(address)}&limit=1&appid=${API_KEY}`;
    const response = await axios.get(geocodeUrl);

    if (response.data && response.data.length > 0) {
      const { lat, lon, name } = response.data[0];
      return { lat, lon, locationName: name };
    }
    return null;
  } catch (error) {
    // console.error('주소 좌표 변환 실패:', error);
    return null;
  }
};

// 풍향을 텍스트로 변환하는 함수
const getWindDirection = (degrees) => {
  if (degrees === undefined || degrees === null) return '정보 없음';

  const directions = [
    { min: 0, max: 11.25, text: '북풍' },
    { min: 11.25, max: 33.75, text: '북북동풍' },
    { min: 33.75, max: 56.25, text: '북동풍' },
    { min: 56.25, max: 78.75, text: '동북동풍' },
    { min: 78.75, max: 101.25, text: '동풍' },
    { min: 101.25, max: 123.75, text: '동남동풍' },
    { min: 123.75, max: 146.25, text: '남동풍' },
    { min: 146.25, max: 168.75, text: '남남동풍' },
    { min: 168.75, max: 191.25, text: '남풍' },
    { min: 191.25, max: 213.75, text: '남남서풍' },
    { min: 213.75, max: 236.25, text: '남서풍' },
    { min: 236.25, max: 258.75, text: '서남서풍' },
    { min: 258.75, max: 281.25, text: '서풍' },
    { min: 281.25, max: 303.75, text: '서북서풍' },
    { min: 303.75, max: 326.25, text: '북서풍' },
    { min: 326.25, max: 348.75, text: '북북서풍' },
    { min: 348.75, max: 360, text: '북풍' }
  ];

  const direction = directions.find(dir => degrees >= dir.min && degrees < dir.max);
  return direction ? direction.text : '북풍';
};

// 좌표를 주소로 변환하는 함수 (Reverse Geocoding)
const getAddressFromCoordinates = async (lat, lon) => {
  try {
    const response = await axios.get(
      `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${API_KEY}`
    );

    if (response.data && response.data.length > 0) {
      const location = response.data[0];
      // 한국어 도시명 매핑
      const cityMapping = {
        'Seoul': '서울',
        'Busan': '부산',
        'Incheon': '인천',
        'Daegu': '대구',
        'Daejeon': '대전',
        'Gwangju': '광주',
        'Ulsan': '울산',
        'Suwon': '수원',
        'Goyang': '고양',
        'Yongin': '용인',
        'Seongnam': '성남'
      };

      const cityName = location.name;
      const koreanCity = cityMapping[cityName] || cityName;

      return {
        cityName: cityName,
        cityKorean: koreanCity,
        locationName: koreanCity
      };
    }
    return null;
  } catch (error) {
    // console.error('좌표 → 주소 변환 실패:', error);
    return null;
  }
};

// 사용자 현재 위치를 가져오는 함수
const getCurrentLocation = async () => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation이 지원되지 않습니다.'));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;

        // 좌표를 주소로 변환 시도
        const addressInfo = await getAddressFromCoordinates(latitude, longitude);

        resolve({
          lat: latitude,
          lon: longitude,
          locationName: addressInfo?.locationName || '현재 위치',
          cityName: addressInfo?.cityName || 'Current Location',
          cityKorean: addressInfo?.cityKorean || '현재 위치'
        });
      },
      (error) => {
        console.error('위치 정보 가져오기 실패:', error);
        reject(error);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000 // 5분간 캐시
      }
    );
  });
};

function WeatherBox() {
  const { user } = useAuth();

  // localStorage에서 이전 날씨 데이터 복원 (즉시 표시용)
  const [weather, setWeather] = useState(() => {
    try {
      const savedWeather = localStorage.getItem('lastWeatherData');
      return savedWeather ? JSON.parse(savedWeather) : null;
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let timeoutId;

    const fetchWeather = async () => {
      // 캐시 키 생성 (농장 주소 기반)
      const farmAddr = user?.selectedFarm?.farmAddr;
      const cacheKey = farmAddr || 'default_location';

      // 캐시된 데이터 확인
      const cachedData = weatherCache.get(cacheKey);
      if (cachedData && (Date.now() - cachedData.timestamp) < WEATHER_CACHE_DURATION) {
        // console.log('캐시된 날씨 데이터 사용');
        setWeather(cachedData.data);
        setLoading(false);
        return;
      }

      // console.log('새로운 날씨 데이터 요청:', new Date().toLocaleTimeString());
      setLoading(true);

      try {
        let weatherUrl;
        let locationName = '서울';
        let currentWeatherUrl; // fallback용

        if (farmAddr) {
          // 1. 농장 주소가 있으면 농장 위치 기반 날씨
          //console.log('농장 주소:', farmAddr);
          const coordinates = await getCoordinatesFromAddress(farmAddr);

          if (coordinates) {
            weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${coordinates.lat}&lon=${coordinates.lon}&units=metric&appid=${API_KEY}`;
            currentWeatherUrl = weatherUrl;
            locationName = coordinates.locationName;
            //console.log('농장 좌표 변환 성공:', coordinates);
          } else {
            // 농장 주소 변환 실패 시 현재 위치로 fallback
            //console.log('농장 주소 변환 실패, 현재 위치로 시도...');
            try {
              const currentLocation = await getCurrentLocation();
              weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${currentLocation.lat}&lon=${currentLocation.lon}&units=metric&appid=${API_KEY}`;
              currentWeatherUrl = weatherUrl;
              locationName = currentLocation.locationName;
              // 현재 위치 정보를 저장해서 나중에 사용
              window.currentLocationInfo = currentLocation;
              //console.log('현재 위치 사용:', currentLocation);
            } catch (locationError) {
              //console.log('현재 위치 실패, 기본 서울 날씨 사용');
              weatherUrl = `https://api.openweathermap.org/data/2.5/weather?q=Seoul&units=metric&appid=${API_KEY}`;
              currentWeatherUrl = weatherUrl;
              locationName = '서울';
            }
          }
        } else {
          // 2. 농장 주소가 없으면 현재 위치 기반 날씨
          //console.log('농장 주소 없음, 현재 위치로 날씨 가져오기...');
          try {
            const currentLocation = await getCurrentLocation();
            weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${currentLocation.lat}&lon=${currentLocation.lon}&units=metric&appid=${API_KEY}`;
            currentWeatherUrl = weatherUrl;
            locationName = currentLocation.locationName;
            // 현재 위치 정보를 저장해서 나중에 사용
            window.currentLocationInfo = currentLocation;
            //console.log('현재 위치 사용:', currentLocation);
          } catch (locationError) {
            //console.log('현재 위치 실패:', locationError.message);
            //console.log('기본 서울 날씨 사용');
            weatherUrl = `https://api.openweathermap.org/data/2.5/weather?q=Seoul&units=metric&appid=${API_KEY}`;
            currentWeatherUrl = weatherUrl;
            locationName = '서울';
          }
        }

        // 현재 날씨만 가져오기 (예보는 필요시에만)
        const currentResponse = await axios.get(weatherUrl);

        // 예보 데이터는 선택적으로 가져오기 (강수확률이 필요한 경우만)
        let forecastData = null;
        try {
          const forecastResponse = await axios.get(weatherUrl.replace('/weather?', '/forecast?'));
          forecastData = forecastResponse.data;
        } catch (forecastError) {
          // console.log('예보 데이터 로딩 실패, 현재 날씨만 표시');
        }

        const currentData = currentResponse.data;

        // 예보 데이터가 있으면 최저/최고 기온과 강수확률 추출
        let tempMin = currentData.main.temp_min || currentData.main.temp;
        let tempMax = currentData.main.temp_max || currentData.main.temp;
        let precipitationProbability = 0; // 강수 확률

        if (forecastData && forecastData.list) {
          const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
          const todayForecasts = forecastData.list.filter(item =>
            item.dt_txt.startsWith(today)
          );

          if (todayForecasts.length > 0) {
            const temps = todayForecasts.map(item => item.main.temp);
            tempMin = Math.min(...temps, currentData.main.temp);
            tempMax = Math.max(...temps, currentData.main.temp);

            // 가장 가까운 시간의 강수 확률 가져오기
            const nextForecast = todayForecasts[0]; // 가장 가까운 예보
            precipitationProbability = nextForecast.pop ? Math.round(nextForecast.pop * 100) : 0;
          }
        }

        const englishCondition = currentData.weather[0].main;
        const koreanCondition =
          weatherKoreanMap[englishCondition] || englishCondition;

        const cityName = currentData.name;
        let cityKorean = cityKoreanMap[cityName] || locationName;

        // 현재 위치 기반인 경우 저장된 정보 사용
        if (window.currentLocationInfo && window.currentLocationInfo.cityKorean) {
          cityKorean = window.currentLocationInfo.cityKorean;
        }

        const weatherData = {
          cityName,
          cityKorean,
          farmAddr: farmAddr || '서울시',
          temp: currentData.main.temp,
          tempMin: tempMin,
          tempMax: tempMax,
          humidity: currentData.main.humidity,
          wind: currentData.wind.speed,
          windDirection: getWindDirection(currentData.wind.deg),
          rain: currentData.rain?.['1h'] || 0,
          precipitationProbability: precipitationProbability,
          condition: koreanCondition,
          iconCode: currentData.weather[0].icon,
          iconUrl: getWeatherIcon(currentData.weather[0].icon),
        };

        // 캐시에 저장
        weatherCache.set(cacheKey, {
          data: weatherData,
          timestamp: Date.now()
        });

        // localStorage에도 저장 (다음 방문 시 즉시 표시용)
        try {
          localStorage.setItem('lastWeatherData', JSON.stringify(weatherData));
        } catch (error) {
          // console.log('날씨 데이터 localStorage 저장 실패');
        }

        setWeather(weatherData);
      } catch (err) {
        console.error('날씨 정보를 가져오는 중 오류 발생:', err);

        // 예보 API 실패 시 현재 날씨만으로라도 표시
        try {
          const response = await axios.get(currentWeatherUrl);
          const data = response.data;

          const englishCondition = data.weather[0].main;
          const koreanCondition = weatherKoreanMap[englishCondition] || englishCondition;
          const cityName = data.name;
          const cityKorean = cityKoreanMap[cityName] || locationName;

          setWeather({
            cityName,
            cityKorean,
            farmAddr: farmAddr || '서울시',
            temp: data.main.temp,
            tempMin: data.main.temp_min,
            tempMax: data.main.temp_max,
            humidity: data.main.humidity,
            wind: data.wind.speed,
            windDirection: getWindDirection(data.wind.deg),
            rain: data.rain?.['1h'] || 0,
            condition: koreanCondition,
            iconCode: data.weather[0].icon,
            iconUrl: getWeatherIcon(data.weather[0].icon),
          });
        } catch (fallbackErr) {
          // console.error('기본 날씨 정보도 가져오기 실패:', fallbackErr);
          // 완전 실패 시 기본 정보 표시
          setWeather({
            cityKorean: '날씨 정보 없음',
            farmAddr: farmAddr || '위치 정보 없음',
            temp: '--',
            condition: '정보 없음',
            iconUrl: null
          });
        }
      } finally {
        setLoading(false);
        // 다음 호출 예약 (10분 후)
        timeoutId = setTimeout(fetchWeather, 600000);
      }
    };

    fetchWeather(); // 초기 한 번 호출

    return () => {
      clearTimeout(timeoutId); // 언마운트 시 타이머 정리
    };
  }, [user?.selectedFarm?.farmAddr]); // farmAddr 변경 시 재호출

  if (loading || !weather)
    return (
      <div className="weather-box">
        <Loader size="text-lg" message="날씨 불러오는 중..." />
      </div>
    );

  // 위치 표시 텍스트 생성
  const getLocationDisplay = () => {
    if (weather.farmAddr && weather.farmAddr !== '서울시') {
      // 농장 주소가 있으면 간단하게 표시
      const parts = weather.farmAddr.split(' ');
      if (parts.length >= 3) {
        return `${parts[0]} ${parts[1]} ${parts[2]}`; // 농장 아이콘
      } else if (parts.length >= 2) {
        return `${parts[0]} ${parts[1]}`;
      }
      return `${weather.farmAddr}`;
    } else if (weather.cityKorean === '현재 위치' || weather.locationName === '현재 위치') {
      // 현재 위치 기반인 경우
      return `현재 위치 (${weather.cityKorean})`;
    }
    return `${weather.cityKorean}`; // 기본값
  };

  return (
    <div className="weather-box">
      <div className="weather-top">
        <div className="weather-location">
          {getLocationDisplay()}
        </div>

        <div className="weather-condition">
          {weather.iconUrl && (
            <img src={weather.iconUrl} alt={weather.condition} className="w-6 h-6" />
          )}
          <span>{weather.condition}</span>
          <span>·</span>
          <span>{typeof weather.temp === 'number' ? `${Math.round(weather.temp)}°` : weather.temp}</span>
        </div>
      </div>
      
      <div className="weather-btm">
        {typeof weather.tempMin === 'number' && typeof weather.tempMax === 'number' && (
          <div className="weather-range">
            최저 {Math.round(weather.tempMin)}° · 최고 {Math.round(weather.tempMax)}°
          </div>
        )}

        {(weather.humidity || weather.wind || weather.rain !== undefined) && (
          <div className="weather-details flex gap-1">
            {weather.rain !== undefined && (
              <div className="flex items-center gap-1">
                강수
                <span className="flex-none">
                  {weather.precipitationProbability}% / {weather.rain} mm
                </span>
              </div>
            )}          
            <span>·</span>
            {weather.humidity && (
              <div className="flex items-center gap-1">
                <span>습도</span>
                <span className="flex-none">{weather.humidity}%</span>
              </div>
            )}
            <span>·</span>
            {weather.wind && (
              <div className="flex items-center gap-1">
                <span className="flex-none">
                  {weather.windDirection} {Math.round(weather.wind * 10) / 10} m/s
                </span>
              </div>
            )}
          </div>
         )}
      </div>

    </div>
  );
}

export default WeatherBox;
