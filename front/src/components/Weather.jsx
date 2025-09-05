import { useWeather } from '../contexts/WeatherContext';

export default function Weather() {
  const { weather, loading, lastFetchedAt } = useWeather() || {};

  if (!weather && loading) {
    return <div className='weather-box'>날씨 불러오는 중...</div>;
  }
  if (!weather) {
    return <div className='weather-box'>날씨 정보 없음</div>;
  }

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

  const getLocationDisplay = () => {
    if (weather.farmAddr && weather.farmAddr !== '서울') {
      const parts = weather.farmAddr.split(' ');
      if (parts.length >= 3) return `${parts[0]} ${parts[1]} ${parts[2]}`;
      if (parts.length >= 2) return `${parts[0]} ${parts[1]}`;
      return weather.farmAddr;
    }
    return weather.cityKorean;
  };

  return (
    <div className='weather-box'>
      <div className='weather-top'>
        <div className='weather-location'>{getLocationDisplay()}</div>

        <div className='weather-condition'>
          {weather.iconUrl && (
            <img
              src={weather.iconUrl}
              alt={weather.condition}
              className='w-6 h-6'
            />
          )}
          <span>{weather.condition}</span>
          <span>·</span>
          <span>
            {typeof weather.temp === 'number'
              ? `${Math.round(weather.temp)}°`
              : weather.temp}
          </span>
        </div>
      </div>

      <div className='weather-btm'>
        {typeof weather.tempMin === 'number' &&
          typeof weather.tempMax === 'number' && (
            <div className='weather-range'>
              최저 {Math.round(weather.tempMin)}° · 최고{' '}
              {Math.round(weather.tempMax)}°
            </div>
          )}

        {(weather.humidity || weather.wind || weather.rain !== undefined) && (
          <div className='weather-details flex gap-1'>
            {weather.rain !== undefined && (
              <div className='flex items-center gap-1'>
                강수
                <span className='flex-none'>
                  {weather.precipitationProbability ?? '-'}% / {weather.rain} mm
                </span>
              </div>
            )}
            <span>·</span>
            {weather.humidity && (
              <div className='flex items-center gap-1'>
                <span>습도</span>
                <span className='flex-none'>{weather.humidity}%</span>
              </div>
            )}
            <span>·</span>
            {weather.wind && (
              <div className='flex items-center gap-1'>
                <span className='flex-none'>
                  {weather.windDirection} {Math.round(weather.wind * 10) / 10}{' '}
                  m/s
                </span>
              </div>
            )}
          </div>
        )}

        <div className='last-update'>
          마지막 업데이트 :
          {lastFetchedAt && <span>{getTimeAgo(lastFetchedAt)}</span>}
        </div>
      </div>
    </div>
  );
}
