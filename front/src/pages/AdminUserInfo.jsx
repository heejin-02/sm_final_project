// src/pages/AdminUserInfo.jsx
// 회원 상세 정보
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getUserDetail } from '../api/admin';
import Loader from '../components/Loader';

export default function AdminUserInfo() {
  const navigate = useNavigate();
  const { userPhone } = useParams();
  const [userDetailList, setUserDetailList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedUserName, setEditedUserName] = useState('');

  // 회원 상세 정보 가져오기
  const fetchUserDetail = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await getUserDetail(userPhone);
      const userDetail = response.data || [];

      setUserDetailList(userDetail);

      // 편집 상태 초기화
      if (userDetail.length > 0) {
        setEditedUserName(userDetail[0].userName);
      }

    } catch (error) {
      console.error('회원 상세 정보 조회 실패:', error);
      setError('데이터 요청이 실패했습니다. 서버 연결을 확인해주세요.');
      setUserDetailList([]);
    } finally {
      setLoading(false);
    }
  };

  // 목록으로 돌아가기
  const handleGoBack = () => {
    navigate('/admin');
  };

  // 회원 삭제
  const handleDeleteUser = () => {
    if (window.confirm(`정말로 ${userDetailList[0]?.userName} 회원을 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.`)) {
      // TODO: 실제 API 호출로 회원 삭제
      console.log('회원 삭제:', userPhone);
      alert('회원이 삭제되었습니다.');
      navigate('/admin');
    }
  };

  // 농장 상세 페이지로 이동
  const handleFarmClick = (farmIdx) => {
    const selectedFarm = userDetailList.find(farm => farm.farmIdx === farmIdx);
    const userInfo = userDetailList[0]; // 첫 번째 항목에서 회원 정보 추출

    navigate(`/admin/farm/${farmIdx}`, {
      state: {
        userInfo: {
          userName: userInfo.userName,
          userPhone: userInfo.userPhone,
          joinedAt: userInfo.joinedAt
        },
        farmInfo: selectedFarm,
        mode: 'edit'
      }
    });
  };

  // 농장 추가 페이지로 이동
  const handleAddFarm = () => {
    const userInfo = userDetailList[0]; // 첫 번째 항목에서 회원 정보 추출

    navigate('/admin/farm/create', {
      state: {
        userInfo: {
          userName: userInfo.userName,
          userPhone: userInfo.userPhone,
          joinedAt: userInfo.joinedAt
        },
        mode: 'create'
      }
    });
  };

  // 수정 모드 시작
  const handleStartEdit = () => {
    setIsEditing(true);
  };

  // 수정 취소
  const handleCancelEdit = () => {
    setIsEditing(false);
    // 원래 값으로 되돌리기
    if (userDetailList.length > 0) {
      setEditedUserName(userDetailList[0].userName);
    }
  };

  // 수정 저장
  const handleSaveEdit = async () => {
    // 이름 유효성 검사
    if (!editedUserName.trim()) {
      alert('이름을 입력해주세요.');
      return;
    }

    if (editedUserName.trim().length < 2) {
      alert('이름은 2글자 이상 입력해주세요.');
      return;
    }

    try {
      // TODO: 실제 API 호출로 회원 이름 수정
      console.log('회원 이름 수정:', editedUserName.trim());

      // 임시로 로컬 상태 업데이트
      const updatedList = userDetailList.map(user => ({
        ...user,
        userName: editedUserName.trim()
      }));
      setUserDetailList(updatedList);

      setIsEditing(false);
      alert('회원 정보가 수정되었습니다.');
    } catch (error) {
      // console.error('회원 정보 수정 실패:', error);
      alert('수정에 실패했습니다.');
    }
  };

  useEffect(() => {
    if (userPhone) {
      fetchUserDetail();
    }
  }, [userPhone]);

  if (loading) {
    return (
      <div className="section flex items-center justify-center">
        <Loader message="회원 상세 정보를 불러오는 중..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="inner inner_1080">
          <h1 className="tit-head">회원 상세 정보</h1>
          <div className="flex items-center justify-end mb-6">
            <button
              onClick={handleGoBack}
              className="btn btn-secondary"
            >
              목록으로
            </button>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <div className="text-red-600 text-lg font-medium mb-2">
              ⚠️ 오류 발생
            </div>
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={() => fetchUserDetail()}
              className="btn btn-accent"
            >
              다시 시도
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!userDetailList || userDetailList.length === 0) {
    return (
      <div className="section">
        <div className="inner inner_1080">
          <h1 className="tit-head">회원 상세 정보</h1>
          <div className="flex justify-between items-center mb-4">
            <h2 className="tit">회원 정보를 찾을 수 없습니다</h2>
            <button
              onClick={handleGoBack}
              className="btn btn-secondary"
            >
              목록으로
            </button>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <p className="text-yellow-700">해당 전화번호로 등록된 회원 정보가 없습니다.</p>
          </div>
        </div>
      </div>
    );
  }

  // 첫 번째 농장 정보에서 회원 기본 정보 추출
  const userInfo = userDetailList[0];

  return (
    <div className="section">
      <div className="inner inner_1080">
        <h1 className="tit-head">회원 상세 정보</h1>
				<div className="flex items-center justify-end mb-6">
					<button
						onClick={handleGoBack}
						className="btn btn-secondary"
					>
						목록으로
					</button>		
				</div>		

        {/* 회원 기본 정보 카드 */}
        <div className="admForm">
          <div className="admForm-header">
            <h3>회원 기본 정보</h3>
            <div className="btn-group">
              {!isEditing ? (
                <button
                  onClick={handleStartEdit}
                  className="btn btn-sm btn-primary"
                >
                  수정
                </button>
              ) : (
                <>
                  <button
                    onClick={handleSaveEdit}
                    className="btn btn-sm btn-accent"
                  >
                    저장
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="btn btn-sm btn-secondary"
                  >
                    취소
                  </button>
                </>
              )}
              <button
                onClick={handleDeleteUser}
                className="btn btn-sm btn-secondary text-red-600 hover:bg-red-50"
              >
                삭제
              </button>
            </div>
          </div>

          <div className="admForm-ul">
            <div className="input-group flex-06">
              <label>이름</label>
              {!isEditing ? (
                <input
                  className="input"
                  value={userInfo.userName || ''}
                  readOnly
                />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-600">이름</label>
              {!isEditing ? (
                <div className="text-lg font-medium text-gray-900 bg-gray-50 px-4 py-3 rounded-lg">
                  {userInfo.userName}
                </div>
              ) : (
                <div>
                  <input
                    type="text"
                    value={editedUserName}
                    onChange={(e) => setEditedUserName(e.target.value)}
                    className={`input ${
                      !editedUserName.trim() || editedUserName.trim().length < 2
                        ? 'border-red-300 focus:border-red-500'
                        : 'border-gray-300 focus:border-blue-500'
                    }`}
                    placeholder="이름을 입력하세요 (2글자 이상)"
                  />
                  {(!editedUserName.trim() || editedUserName.trim().length < 2) && (
                    <p className="text-red-500 text-sm mt-1">
                      {!editedUserName.trim() ? '이름을 입력해주세요.' : '이름은 2글자 이상 입력해주세요.'}
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="input-group flex-08">
              <label>아이디(휴대폰번호)</label>
              <input
                className="input"
                value={userInfo.userPhone || ''}
                readOnly
              />
            </div>

            <div className="input-group flex-1">
              <label>가입날짜</label>
              <input
                className="input"
                value={userInfo.joinedAt || ''}
                readOnly
              />
            </div>
          </div>
        </div>
        
        <div className='max-w-5xl mx-auto'>
          {/* 농장 추가 버튼 */}
          <div className="flex justify-end mb-4">
            <button
              onClick={handleAddFarm}
              className="btn btn-primary"
            >
              농장 추가
            </button>
          </div>

          {/* 농장 목록 테이블 */}
          <div className="table-wrap">
            <div className="overflow-x-auto">
              <table className="table">
                <colgroup>
                  <col width="10%"/>
                  <col width="15%"/>
                  <col width="20%"/>
                  <col width="30%"/>
                  <col width="25%"/>
                </colgroup>
                <thead>
                  <tr>
                    <th>번호</th>
                    <th>재배 작물</th>
                    <th>농장 이름 / 하우스</th>
                    <th>농장 주소</th>
                    <th>농장 번호</th>
                  </tr>
                </thead>
                <tbody>
                  {userDetailList.length > 0 && userDetailList.some(farm => farm.farmName) ? (
                    userDetailList
                      .filter(farm => farm.farmName) // 농장 이름이 있는 것만 필터링
                      .sort((a, b) => a.farmIdx - b.farmIdx) // farmIdx 낮은 순으로 정렬
                      .map((farm, index) => (
                        <tr
                          key={farm.farmIdx || `farm-${index}`}
                          onClick={() => handleFarmClick(farm.farmIdx)}
                          className="cursor-pointer hover:bg-gray-50 transition-colors"
                        >
                          <td data-farm-idx={farm.farmIdx}>{index + 1}</td>
                          <td>{farm.farmCrops || '-'}</td>
                          <td>{farm.farmName}</td>
                          <td>{farm.farmAddr || '-'}</td>
                          <td>{farm.farmPhone || '-'}</td>
                        </tr>
                      ))
                  ) : (
                    <tr>
                      <td colSpan="7" className="text-center text-gray-500 py-8">
                        등록된 농장 정보가 없습니다.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}