// src/components/LoginForm.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function LoginForm() {
  return(
	 <form className="form login-form" action="" method='post'>
		<ul className="form-ul">
			<li>
				<span className="frm-label">아이디</span>
				<input type="text" placeholder="휴대폰번호를 입력해주세요" className="frm-input" id="" name="" value=""/>
			</li>
			<li>
				<span className="frm-label">비밀번호</span>
				<input type="password" placeholder="비밀번호를 입력해주세요" className="frm-input" id="" name=""/>
			</li>
		</ul>
		<button type="submit" className="btn-submit">로그인</button>		
	 </form>
  )

}

export default LoginForm;