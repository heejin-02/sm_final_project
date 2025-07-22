// src/components/LoginForm.jsx
import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function LoginForm({ onLogin }) {

	const [id, setId] = useState('');           // ① 아이디 상태 추가
	const [pw, setPw] = useState('');           // ② 비밀번호 상태 추가
	const navigate = useNavigate();             // ③ useNavigate 훅 가져오기


  // ④ 폼 제출 핸들러
  const handleSubmit = async e => {
    e.preventDefault();                       // → 기본 리다이렉트 방지
    const resp = await fetch('http://localhost:8095/web/api/auth/login', {
      method: 'POST',
      credentials: 'include',
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify({ userPhone: id, userPw: pw })
    })
    if (!resp.ok) {
      alert('로그인 실패')
      return
    }
    const { role } = await resp.json()

    onLogin(role) // ← App의 setRole 호출
    if (role === 'admin') navigate('/admin')
    else navigate('/select-farm')
  }

  return(
	 <form className="login-form" onSubmit={handleSubmit}>
		<ul className="form-ul">
			<li>
				<span htmlFor="userPhone" className="frm-label">아이디</span>
				<input
					id="userPhone"
					name="userPhone"
					type="text"
					placeholder="휴대폰번호를 입력해주세요"
					className="frm-input"
					value={id}
					onChange={e => setId(e.target.value)}    // → 값 변경 반영
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
					onChange={e => setPw(e.target.value)}    // → 값 변경 반영
				/>
			</li>
		</ul>
		<button type="submit" className="btn-submit">로그인</button>		
	 </form>
  )

}