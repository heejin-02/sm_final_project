// src/pages/SelectFarm
// 로그인 후 계정에 등록된 농장이 1개 초과면 해당 페이지,
// 아니면 바로 MainFarm으로 가시면 됩니다!
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { getUserFarms } from '../api/auth';
import bgImg from '/images/bg_home.jpg';

export default function SelectFarm() {

  const { user, selectFarm } = useAuth();
  const navigate = useNavigate();

  const [farms, setFarms] = useState([]);      // 렌더링용 농장 리스트
  const [loading, setLoading] = useState(true); // 로딩 플래그  

  useEffect(() => {
  if (!user?.phone) return;

  setLoading(true);
  getUserFarms(user.phone)
    .then(res => {
      setFarms(res.data);
    })
    .catch(err => {
      console.error('농장 불러오기 실패', err);
      setFarms([]);
    })
    .finally(() => {
      setLoading(false);
    });
  }, [user?.phone]);

  // 농장이 1개 이하라면 자동 선택 & 네비게이트
  useEffect(() => {
    if (!loading && farms.length === 1) {
      selectFarm(farms[0]);
      navigate(`/mainfarm/${farms[0].id}`, { replace: true });
    }
  }, [loading, farms, selectFarm, navigate]);

  if (loading) {
    return <div className="text-center p-8">로딩 중...</div>;
  }

  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center flex flex-col justify-center gap-6">
        <h2 className="font-bold text-3xl mb-4">관리할 농장을 선택해주세요</h2>
        {farms.length === 0 ? (
          <p>등록된 농장이 없습니다.</p>
        ) : (
          <ul className="flex flex-wrap justify-center gap-2">
            {farms.map(farm => (
              <li
                key={farm.id}
                className="farmList-item cursor-pointer"
                onClick={() => {
                  selectFarm(farm);
                  navigate(`/mainfarm/${farm.id}`);
                }}
              >
                {farm.name}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )

}
