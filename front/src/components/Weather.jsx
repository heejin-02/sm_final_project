// openweathermap API 사용

import { useEffect, useState } from 'react';
import axios from 'axios';
import Loader from './Loader';
import {
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

function WeatherBox() {
  const [weather, setWeather] = useState(null);

  useEffect(() => {
    let timeoutId;

    const fetchWeather = async () => {
      console.log('날씨 데이터 요청:', new Date().toLocaleTimeString());
      try {
        const url = `https://api.openweathermap.org/data/2.5/weather?q=${CITY}&units=metric&appid=${API_KEY}`;
        const response = await axios.get(url);
        const data = response.data;

        const englishCondition = data.weather[0].main;
        const koreanCondition =
          weatherKoreanMap[englishCondition] || englishCondition;

        const cityName = data.name;
        const cityKorean = cityKoreanMap[cityName] || cityName;

        setWeather({
          cityName,
          cityKorean,
          temp: data.main.temp,
          tempMin: data.main.temp_min,
          tempMax: data.main.temp_max,
          humidity: data.main.humidity,
          wind: data.wind.speed,
          rain: data.rain?.['1h'] || 0,
          condition: koreanCondition,
          iconCode: data.weather[0].icon, // 아이콘 코드 추가
          iconUrl: getWeatherIcon(data.weather[0].icon), // 아이콘 URL 추가
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
      <div className="flex items-center justify-center h-16">
        <Loader size="text-lg" message="날씨 불러오는 중..." />
      </div>
    );

  return (
    <div className="">
      <div className="font-bold mb-2">
        {weather.cityKorean} 마포구
      </div>

      <div className="flex items-center gap-1">
        <img
          src={weather.iconUrl}
          alt={weather.condition}
          className="w-8 h-8"
        />
        <span>
          {weather.condition} · {weather.temp}°C
        </span>
      </div>

      <div className="text-sm text-gray-600 mb-1">
        최저 {weather.tempMin}°C · 최고 {weather.tempMax}°C
      </div>

      <div className="flex items-center">
        <div className="flex items-center gap-1">
          {/* 습도 */}
          <WiHumidity size={22} />
          <span className="flex-none">{weather.humidity}%</span>
        </div>

        <div className="flex items-center gap-1">
          {/* 풍속 */}
          <WiStrongWind size={22} />
          <span className="flex-none">{weather.wind} m/s</span>
        </div>

        <div className="flex items-center gap-1">
          {/* 강수량 */}
          <WiRaindrops size={22} />
          <span className="flex-none">{weather.rain} mm</span>
        </div>        
      </div>

    </div>
  );
}

export default WeatherBox;
