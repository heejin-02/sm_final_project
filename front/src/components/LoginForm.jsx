// src/components/LoginForm.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
 

export default function LoginForm() {
	const [id, setId] = useState('');
	const [pw, setPw] = useState('');
	const navigate = useNavigate();
	const { login } = useAuth();

	const handleSubmit = async e => {
    e.preventDefault();
    try {
      const response = await login(id.trim(), pw);  // AuthContext의 login 사용
      const userData = response.data;
      navigate(userData.role === 'admin' ? '/admin' : '/selectFarm', { replace: true });
    } catch (err) {
      console.error(err);
      alert('로그인 실패: 아이디/비밀번호를 확인하세요.');
    }
  };
  return (
    <form className="login-form" onSubmit={handleSubmit}>
      {/* <h2 className="font-semibold text-2xl text-center mb-4">로그인</h2> */}
      <ul className="form-ul text-left">
        <li>
          <span htmlFor="userPhone" className="frm-label">아이디</span>
          <input
            id="userPhone"
            name="userPhone"
            type="text"
            placeholder="휴대폰번호를 입력해주세요"
            className="input"
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
            className="input"
            value={pw}
            onChange={e => setPw(e.target.value)}
          />
        </li>
      </ul>
      <button type="submit" className="btn btn-lg btn-primary w-full mt-4">로그인</button>
    </form>
  );
}
