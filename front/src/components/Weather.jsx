import { useWeather } from '../contexts/WeatherContext';

export default function Weather() {
  const { weather, loading, lastFetchedAt } = useWeather() || {};

  if (!weather && loading) {
    return <div className='weather-box'>날씨 불러오는 중...</div>;
  }
  if (!weather) {
    return <div className='weather-box'>날씨 정보 없음</div>;
  }

  // MainFarm과 동일한 형태의 "몇 분 전" 함수
  const getTimeAgo = (timestamp) => {
    if (!timestamp) return '업데이트 없음';

    const now = new Date();
    const diff = now - new Date(timestamp);
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));

    if (minutes < 1) return '방금 전';
    if (minutes < 60) return `${minutes}분 전`;
    if (hours < 24) return `${hours}시간 전`;
    return `${Math.floor(hours / 24)}일 전`;
  };

  return (
    <div className='weather-box'>
      <div className='weather-top'>
        <div className='weather-location'>{weather.cityKorean}</div>
        <div className='weather-condition'>
          {weather.iconUrl && (
            <img
              src={weather.iconUrl}
              alt={weather.condition}
              className='w-6 h-6'
            />
          )}
          <span>{weather.condition}</span> ·{' '}
          <span>{Math.round(weather.temp)}°</span>
        </div>
      </div>

      <div className='weather-btm'>
        <div className='weather-range'>
          최저 {Math.round(weather.tempMin)}° · 최고{' '}
          {Math.round(weather.tempMax)}°
        </div>
        <div className='weather-details'>
          습도 {weather.humidity}% · 바람 {weather.wind} m/s
        </div>

        <div className='last-update'>
          마지막 업데이트 :
          {lastFetchedAt && <span>{getTimeAgo(lastFetchedAt)}</span>}
        </div>
      </div>
    </div>
  );
}
