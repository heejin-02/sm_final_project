// src/pages/SelectFarm
// 로그인 후 계정에 등록된 농장이 1개 초과면 해당 페이지,
// 아니면 바로 MainFarm으로 가시면 됩니다!
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import bgImg from '/images/bg_home.jpg';

export default function SelectFarm() {

  const navigate = useNavigate();
  const { user, selectFarm } = useAuth();
  
  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center flex flex-col justify-center gap-6">
        <h2 className="font-bold text-3xl mb-4">관리할 농장을 선택해주세요</h2>
        <ul className="flex flex-wrap justify-center gap-2">
          {user.farms.map(farm => (
            <li key={farm.id} className="farmList-item"
            onClick={() => {
              selectFarm(farm);
              navigate(`/mainfarm/${farm.id}`);
            }}
            >
              {farm.name}
            </li>  
          ))}     
        </ul>
      </div>
    </div>
  )

}
