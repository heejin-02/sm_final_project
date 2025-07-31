// openweathermap API 사용

import { useEffect, useState } from 'react';
import axios from 'axios';
import Loader from './Loader';
import { useAuth } from '../contexts/AuthContext';
import {
  WiHumidity,
  WiStrongWind,
  WiRaindrops,
} from 'react-icons/wi';

const API_KEY = '019d414c565826322ad2f0b73af0129b';

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
    console.error('주소 좌표 변환 실패:', error);
    return null;
  }
};

// 사용자 현재 위치를 가져오는 함수
const getCurrentLocation = () => {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation이 지원되지 않습니다.'));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        resolve({
          lat: latitude,
          lon: longitude,
          locationName: '현재 위치'
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
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let timeoutId;

    const fetchWeather = async () => {
      console.log('날씨 데이터 요청:', new Date().toLocaleTimeString());
      setLoading(true);

      try {
        const farmAddr = user?.selectedFarm?.farmAddr;
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
            //console.log('🔄 농장 주소 변환 실패, 현재 위치로 시도...');
            try {
              const currentLocation = await getCurrentLocation();
              weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${currentLocation.lat}&lon=${currentLocation.lon}&units=metric&appid=${API_KEY}`;
              currentWeatherUrl = weatherUrl;
              locationName = currentLocation.locationName;
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
            //console.log('현재 위치 사용:', currentLocation);
          } catch (locationError) {
            //console.log('현재 위치 실패:', locationError.message);
            //console.log('기본 서울 날씨 사용');
            weatherUrl = `https://api.openweathermap.org/data/2.5/weather?q=Seoul&units=metric&appid=${API_KEY}`;
            currentWeatherUrl = weatherUrl;
            locationName = '서울';
          }
        }

        // 현재 날씨와 5일 예보를 동시에 가져오기
        const [currentResponse, forecastResponse] = await Promise.all([
          axios.get(weatherUrl),
          axios.get(weatherUrl.replace('/weather?', '/forecast?'))
        ]);

        const currentData = currentResponse.data;
        const forecastData = forecastResponse.data;

        // 오늘 날짜의 예보 데이터에서 최저/최고 기온 추출
        const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
        const todayForecasts = forecastData.list.filter(item =>
          item.dt_txt.startsWith(today)
        );

        let tempMin = currentData.main.temp;
        let tempMax = currentData.main.temp;

        if (todayForecasts.length > 0) {
          const temps = todayForecasts.map(item => item.main.temp);
          tempMin = Math.min(...temps, currentData.main.temp);
          tempMax = Math.max(...temps, currentData.main.temp);
        }

        const englishCondition = currentData.weather[0].main;
        const koreanCondition =
          weatherKoreanMap[englishCondition] || englishCondition;

        const cityName = currentData.name;
        const cityKorean = cityKoreanMap[cityName] || locationName;

        setWeather({
          cityName,
          cityKorean,
          farmAddr: farmAddr || '서울시',
          temp: currentData.main.temp,
          tempMin: tempMin,
          tempMax: tempMax,
          humidity: currentData.main.humidity,
          wind: currentData.wind.speed,
          rain: currentData.rain?.['1h'] || 0,
          condition: koreanCondition,
          iconCode: currentData.weather[0].icon,
          iconUrl: getWeatherIcon(currentData.weather[0].icon),
        });
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
            rain: data.rain?.['1h'] || 0,
            condition: koreanCondition,
            iconCode: data.weather[0].icon,
            iconUrl: getWeatherIcon(data.weather[0].icon),
          });
        } catch (fallbackErr) {
          console.error('기본 날씨 정보도 가져오기 실패:', fallbackErr);
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
      <div className="flex items-center justify-center h-16">
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
    <div className="">
      <div className="font-bold mb-2">
        {getLocationDisplay()}
      </div>

      <div className="flex items-center gap-1">
        {weather.iconUrl && (
          <img
            src={weather.iconUrl}
            alt={weather.condition}
            className="w-8 h-8"
          />
        )}
        <span>
          {weather.condition} · {typeof weather.temp === 'number' ? `${Math.round(weather.temp)}°C` : weather.temp}
        </span>
      </div>

      {typeof weather.tempMin === 'number' && typeof weather.tempMax === 'number' && (
        <div className="text-sm text-gray-600 mb-1">
          최저 {Math.round(weather.tempMin)}°C · 최고 {Math.round(weather.tempMax)}°C
        </div>
      )}

      {(weather.humidity || weather.wind || weather.rain !== undefined) && (
        <div className="flex items-center">
          {weather.humidity && (
            <div className="flex items-center gap-1">
              {/* 습도 */}
              <WiHumidity size={22} />
              <span className="flex-none">{weather.humidity}%</span>
            </div>
          )}

          {weather.wind && (
            <div className="flex items-center gap-1">
              {/* 풍속 */}
              <WiStrongWind size={22} />
              <span className="flex-none">{Math.round(weather.wind * 10) / 10} m/s</span>
            </div>
          )}

          {weather.rain !== undefined && (
            <div className="flex items-center gap-1">
              {/* 강수량 */}
              <WiRaindrops size={22} />
              <span className="flex-none">{weather.rain} mm</span>
            </div>
          )}
        </div>
      )}

    </div>
  );
}

export default WeatherBox;
