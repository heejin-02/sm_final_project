// src/pages/SelectFarm
// 로그인 후 계정에 등록된 농장이 1개 초과면 해당 페이지,
// 아니면 바로 mainFarm으로 가시면 됩니다!
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import bgImg from '/images/bg_home.jpg';
import FarmList from '../components/FarmList';


export default function SelectFarm() {
  return(
    <div className="section bg home bg-[url('/images/bg_home.jpg')] bg-center bg-cover">
      <div className="cont-wrap text-center flex flex-col justify-center gap-6">
        <FarmList/>
      </div>
    </div>
  )

}
