import { useState } from 'react';
import { addUser } from '../api/admin';

export default function AddUserModal({ isOpen, onClose, onSuccess }) {
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

    if (formData.userPw !== userPwChk) {
      alert('비밀번호가 일치하지 않습니다.');
      return;
    }

    if (!formData.userName || !formData.userPhone || !formData.userPw) {
      alert('모든 필드를 입력해주세요.');
      return;
    }

    try {
      setLoading(true);
      await addUser(formData);
      alert('회원 등록이 완료되었습니다.');
      
      // 폼 초기화
      setFormData({ userName: '', userPhone: '', userPw: '' });
      setUserPwChk('');
      
      onSuccess();
      onClose();
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

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 class="title">회원 등록</h3>
          <button onClick={onClose} className="modal-close-btn">×</button>
        </div>
        
        <form onSubmit={handleSubmit}>
          <ul className="form-ul">
            <li>
              <span className="frm-label">이름</span>
              <input
                type="text"
                name="userName"
                value={formData.userName}
                onChange={handleChange}
                placeholder="이름을 입력하세요"
                className="input"
                required
              />
            </li>
            
            <li>
              <span className="frm-label">휴대폰번호</span>
              <input
                type="text"
                name="userPhone"
                value={formData.userPhone}
                onChange={handleChange}
                placeholder="휴대폰번호를 입력하세요"
                className="input"
                required
              />
            </li>
            
            <li>
              <span className="frm-label">비밀번호</span>
              <input
                type="password"
                name="userPw"
                value={formData.userPw}
                onChange={handleChange}
                placeholder="비밀번호를 입력하세요"
                className="input"
                required
              />
            </li>
            
            <li>
              <span className="frm-label">비밀번호 확인</span>
              <input
                type="password"
                value={userPwChk}
                onChange={(e) => setUserPwChk(e.target.value)}
                placeholder="비밀번호를 다시 입력하세요"
                className="input"
                required
              />
            </li>
          </ul>
          
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.5rem' }}>
            <button type="button" onClick={onClose} className="btn btn-gray" style={{ flex: 1 }}>
              취소
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading} style={{ flex: 1 }}>
              {loading ? '등록 중...' : '회원등록'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}