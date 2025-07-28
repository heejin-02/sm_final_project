// pages/AddUser.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { addUser } from '../api/admin';

export default function AddUser() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    userName: '',
    userPhone: '',
    userPw: ''
  });
  const [userPwChk, setUserPwChk] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // 비밀번호 확인 검증
    if (formData.userPw !== userPwChk) {
      alert('비밀번호가 일치하지 않습니다.');
      return;
    }

    // 필수 필드 검증
    if (!formData.userName || !formData.userPhone || !formData.userPw) {
      alert('모든 필드를 입력해주세요.');
      return;
    }

    try {
      setLoading(true);
      const response = await addUser(formData);
      alert('회원 등록이 완료되었습니다.');
      navigate('/admin'); // 성공 시 관리자 페이지로 이동
    } catch (err) {
      console.error(err);
      if (err.response?.status === 400) {
        alert('이미 등록된 회원입니다.');
      } else {
        alert('회원 등록에 실패했습니다.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="section p-6">
			<div className="admin">
				<h1 className="tit">회원 등록</h1>
				<form onSubmit={handleSubmit} className="form-ul">

					<div className="form-item">
						<label htmlFor="userPhone">전화번호(아이디)</label>
						<input
							type="text"
							id="userPhone"
							name="userPhone"
							value={formData.userPhone}
							onChange={handleChange}
							placeholder="010-1234-5678"
							required
						/>
					</div>
					<div className="form-item">
						<label htmlFor="userName">이름</label>
						<input
							type="text"
							id="userName"
							name="userName"
							value={formData.userName}
							onChange={handleChange}
							placeholder="홍길동"
							required
						/>
					</div>					
					<div className="form-item">
						<label htmlFor="userPw">비밀번호</label>
						<input
							type="password"
							id="userPw"
							name="userPw"
							value={formData.userPw}
							onChange={handleChange}
							placeholder="비밀번호를 입력하세요"
							required
						/>
					</div>
					<div className="form-item">
						<label htmlFor="userPwChk">비밀번호 확인</label>
						<input
							type="password"
							id="userPwChk"
							name="userPwChk"
							value={userPwChk}
							onChange={(e) => setUserPwChk(e.target.value)}
							placeholder="비밀번호를 다시 입력하세요"
							required
						/>
					</div>	
					<button
						type="submit"
						className="btn submit-btn"
						disabled={loading}
					>
						{loading ? '등록 중...' : '회원등록'}
					</button>
				</form>
			</div>
    </div>
  );
}