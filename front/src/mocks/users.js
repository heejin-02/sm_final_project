import { TbDeviceIpadX } from "react-icons/tb";

// 목업용 회원 더미 데이터 - 백엔드랑 연결되면 삭제
export const DUMMY_USERS = [
  {
    user_phone: 'admin',
		user_pw : 'admin',
		user_name : '관리자',
  },
  {
    user_phone: '010-0000-0001',
		user_pw : 'pw1',
		user_name : '박용필',
    farms: [
      { farm_idx: 1, farm_name: '토마토 A' },
      { farm_idx: 2, farm_name: '토마토 B' },
			{ farm_idx: 3, farm_name: '토마토 C' },
			{ farm_idx: 4, farm_name: '토마토 D' },
    ]
  },	
];