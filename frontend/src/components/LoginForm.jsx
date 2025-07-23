// src/components/LoginForm.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login as loginAPI, getCurrentUser } from '../api/auth'; // 서버 요청용 login
import { useAuth } from '../contexts/AuthContext'; // 상태 저장용 login
import { DUMMY_FARMS } from '../mocks/farms';

export default function LoginForm({ onLogin }) {
	const [id, setId] = useState('');
	const [pw, setPw] = useState('');
	const navigate = useNavigate();
	const { login } = useAuth(); // ✅ 상태 저장 함수

	const handleSubmit = async e => {
	e.preventDefault();
	try {
		await loginAPI(id, pw); // 1단계: 로그인 요청

		// 2단계: 로그인 성공 후 사용자 정보 요청
		const { data } = await getCurrentUser();

    const userData = {
      name: data.userName, // ✅ 바로 data.userName
      phone: data.userPhone,
      role: data.role,
      farms: DUMMY_FARMS, // ← 여기에 더미 데이터 주입
      selectedFarm: null
    };

		login(userData); // context에 저장
		navigate(userData.role === 'admin' ? '/admin' : '/select-farm', { replace: true });

	} catch (err) {
		console.error('로그인 실패:', err);
		alert('로그인 실패');
	}
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
