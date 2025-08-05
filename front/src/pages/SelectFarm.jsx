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

      // 농장 선택
      selectFarm(farms[0]);

      // 약간의 지연 후 이동 (상태 업데이트 완료 대기)
      setTimeout(() => {
        navigate(`/mainfarm/${farms[0].farmIdx}`, { replace: true });
      }, 10);

    }
  }, [loading, farms.length]); // selectFarm, navigate 의존성 제거

  if (loading) {
    return <div className="text-center p-8">로딩 중...</div>;
  }

  return(
    <div className="select-farm section flex flex-col bg-[url('/images/home_bg.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center items-center justify-center">
        {farms.length === 0 ? (
          <div className="mt-4 space-y-4 text-white">
            <p className='text-2xl '>등록된 농장이 없습니다. <br/>관리자에게 문의하세요.</p>
            <p className='text-2xl font-semibold'>관리자 번호 : 010-109-1009</p>
          </div>
        ) : (
          <div className='white-box'>
            <h2 className="tit-3">관리할 농장을 선택해주세요</h2>
            <ul className="farmList">
              {farms
                .sort((a, b) => a.farmIdx - b.farmIdx) // farmIdx 낮은 순으로 정렬
                .map(farm => (
                <li
                  key={farm.farmIdx}
                  className="farmList-item"
                  onClick={() => {
                    selectFarm(farm);
                    navigate(`/mainfarm/${farm.farmIdx}`);
                  }}
                >
                  {farm.farmName}
                </li>
              ))}
            </ul>              
          </div>
        )}
      </div>
    </div>
  )

}
