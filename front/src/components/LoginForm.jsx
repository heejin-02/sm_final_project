// src/components/LoginForm.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function LoginForm() {
  return(
	 <form className="form">
		<ul className="form-ul">
			<li>
				<span>아이디</span>
				<input type="text" placeholder="아이디" className="frm-input"/>
			</li>
			<li>
				<span>비밀번호</span>
				<input type="password" placeholder="비밀번호" className="frm-input"/>
			</li>
		</ul>
		<button type="submit">로그인</button>		
	 </form>
  )

}

export default LoginForm;