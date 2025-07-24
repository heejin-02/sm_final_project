// src/components/LoginForm.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
// import { login as loginAPI } from '../api/auth'; // 서버 요청용 login
import { useAuth } from '../contexts/AuthContext'; // 상태 저장용 login
import { DUMMY_USERS } from '../mocks/users';
 

export default function LoginForm({ onLogin }) {
	const [id, setId] = useState('');
	const [pw, setPw] = useState('');
	const navigate = useNavigate();
	const { login } = useAuth(); // ✅ 상태 저장 함수

	const handleSubmit = async e => {
    e.preventDefault();
    
    // 1) 백엔드 호출 부분 주석 처리
    // try {
    // 	await loginAPI(id, pw); // 1단계: 로그인 요청

    // 	// 2단계: 로그인 성공 후 사용자 정보 요청
    // 	const { data } = await getCurrentUser();

    //   const userData = {
    //     phone: data.userPhone,
    //     name: data.userName,
    //     role: data.role,
    //     farms: farms,
    //     selectedFarm: null
    //   };

    // 	login(userData); // context에 저장
    // 	navigate(userData.role === 'admin' ? '/admin' : '/select-farm', { replace: true });

    // } catch (err) {
    // 	console.error('로그인 실패:', err);
    // 	alert('로그인 실패');
    // }
    // };

    // 2) DUMMY_USERS 에서 검색
    const user = DUMMY_USERS.find(u =>
      u.user_phone === id.trim() && u.user_pw === pw
    );

    if (!user) {
      alert('아이디 또는 비밀번호가 올바르지 않습니다.');
      return;
    }

    // 3) Context용 유저 데이터 포맷 맞추기
    const userData = {
      phone: user.user_phone,
      name: user.user_name,
      role: user.role || 'user',
      farms: user.farms || [],
      selectedFarm: null,
    };  

    // 4) Context에 저장하고 페이지 이동
    login(userData);
    navigate(
      userData.role === 'admin' ? '/admin' : '/select-farm',
      { replace: true }
    );
  };  

  return (
    <form className="login-form" onSubmit={handleSubmit}>
      <ul className="form-ul text-left">
        <li>
          <span htmlFor="userPhone" className="frm-label">아이디</span>
          <input
            id="userPhone"
            name="userPhone"
            type="text"
            placeholder="휴대폰번호를 입력해주세요"
            className="frm-input"
            value={id}
            onChange={e => setId(e.target.value)}
          />
        </li>
        <li>
          <span htmlFor="userPw" className="frm-label">비밀번호</span>
          <input
            id="userPw"
            name="userPw"
            type="password"
            placeholder="비밀번호를 입력해주세요"
            className="frm-input"
            value={pw}
            onChange={e => setPw(e.target.value)}
          />
        </li>
      </ul>
      <button type="submit" className="btn-submit">로그인</button>
    </form>
  );
}
