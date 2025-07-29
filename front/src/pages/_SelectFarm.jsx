// src/pages/SelectFarm
// 로그인 후 계정에 등록된 농장이 1개 초과면 해당 페이지,
// 아니면 바로 MainFarm으로 가시면 됩니다!
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { getUserFarms } from '../api/auth';

export default function SelectFarm() {

  const { user, selectFarm } = useAuth();
  const navigate = useNavigate();

  const [farms, setFarms] = useState([]);      // 렌더링용 농장 리스트
  const [loading, setLoading] = useState(true); // 로딩 플래그  

  useEffect(() => {
    if (!user?.userPhone) return;

    setLoading(true);
    getUserFarms(user.userPhone)
      .then(res => {
        // 백엔드 응답 구조 그대로 사용
        const farmData = res.data;
        setFarms(farmData);
      })
      .catch(err => {
        console.error('농장 불러오기 실패', err);
        setFarms([]);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [user?.userPhone]);

  // 농장이 1개 이하라면 자동 선택 & 네비게이트
  useEffect(() => {
    if (!loading && farms.length === 1) {
      selectFarm(farms[0]);
      navigate(`/mainfarm/${farms[0].farmIdx}`, { replace: true });
    }
  }, [loading, farms, selectFarm, navigate]);

  if (loading) {
    return <div className="text-center p-8">로딩 중...</div>;
  }

  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center flex flex-col justify-center gap-6">
        <h2 className="font-semibold text-3xl mb-2">관리할 하우스를 선택해주세요</h2>
        {farms.length === 0 ? (
          <p>등록된 하우스가 없습니다.</p>
        ) : (
          <ul className="flex flex-wrap gap-2 w-full max-w-[1200px]">
            {farms.map(farm => (
              <li
                key={farm.farmIdx}
                className="farmList-item cursor-pointer"
                onClick={() => {
                  selectFarm(farm);
                  navigate(`/mainfarm/${farm.farmIdx}`);
                }}
              >
                {farm.farmName}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )

}
