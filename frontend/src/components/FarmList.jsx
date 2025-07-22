// src/components/FarmList.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser } from '../api/auth';

function FarmList() {
	const navigate = useNavigate();
	const [userInfo, setUserInfo] = useState(null);

	useEffect(() => {
	async function fetchUser() {
		try {
		const { data } = await getCurrentUser();
		setUserInfo(data); // 예: { name: '김영희', farmName: '행복농장' }
		} catch (e) {
		console.error('사용자 정보 불러오기 실패:', e);
		}
	}
	fetchUser();
	}, []);

	return(
	 <ul>
		<li>
			<a href="">농장1</a>
		</li>
	 </ul>
  )

}

export default FarmList;