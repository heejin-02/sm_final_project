// openweathermap API 사용

import React, { useEffect, useState } from 'react';
import {
  WiDaySunny,
  WiHumidity,
  WiStrongWind,
  WiRaindrops,
} from 'react-icons/wi';

const API_KEY = '019d414c565826322ad2f0b73af0129b';
const CITY = 'Seoul';

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

function WeatherBox() {
  const [weather, setWeather] = useState(null);

  useEffect(() => {
    let timeoutId;

    const fetchWeather = async () => {
      console.log('날씨 데이터 요청:', new Date().toLocaleTimeString());
      try {
        const url = `https://api.openweathermap.org/data/2.5/weather?q=${CITY}&units=metric&appid=${API_KEY}`;
        const res = await fetch(url);

        if (!res.ok) {
          throw new Error(`API 요청 실패: ${res.status} ${res.statusText}`);
        }

        const data = await res.json();

        const englishCondition = data.weather[0].main;
        const koreanCondition =
          weatherKoreanMap[englishCondition] || englishCondition;

        const cityName = data.name;
        const cityKorean = cityKoreanMap[cityName] || cityName;

        setWeather({
          cityName,
          cityKorean,
          temp: data.main.temp,
          humidity: data.main.humidity,
          wind: data.wind.speed,
          rain: data.rain?.['1h'] || 0,
          condition: koreanCondition,
        });
      } catch (err) {
        console.error('날씨 정보를 가져오는 중 오류 발생:', err);
      } finally {
        // 다음 호출 예약 (10분 후)
        timeoutId = setTimeout(fetchWeather, 600000); // 600,000ms = 10분
      }
    };

    fetchWeather(); // 초기 한 번 호출

    return () => {
      clearTimeout(timeoutId); // 언마운트 시 타이머 정리
    };
  }, []);

  if (!weather)
    return (
      <div className="text-gray-600 animate-pulse">날씨 불러오는 중...</div>
    );

  return (
    <div className="p-4 text-base">
      <div className="font-bold mb-2">
        {weather.cityKorean} 마포구
      </div>

      <div className="flex items-center gap-2 mb-1">
        <WiDaySunny size={24} />
        <span>
          {weather.condition} · {weather.temp}°C
        </span>
      </div>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2 mb-1">
          {/* 습도 */}
          <WiHumidity size={22} />
          <span className="flex-none">{weather.humidity}%</span>
        </div>

        <div className="flex items-center gap-2 mb-1">
          {/* 풍속 */}
          <WiStrongWind size={22} />
          <span className="flex-none">{weather.wind} m/s</span>
        </div>

        <div className="flex items-center gap-2">
          {/* 강수량 */}
          <WiRaindrops size={22} />
          <span className="flex-none">{weather.rain} mm</span>
        </div>        
      </div>

    </div>
  );
}

export default WeatherBox;
