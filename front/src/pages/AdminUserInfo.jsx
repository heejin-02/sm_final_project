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

  // 회원 상세 정보 가져오기
  const fetchUserDetail = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await getUserDetail(userPhone);
      const userDetail = response.data || [];

      setUserDetailList(userDetail);

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
      <div className="section p-6">
        <div className="max-w-7xl mx-auto">
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
        <div className="inner">
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
      <div className="inner">
        <h1 className="tit-head">회원 상세 정보</h1>

        {/* 회원 기본 정보 */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 mb-6 max-w-5xl mx-auto">
          <div className="flex justify-center gap-6">
            <div>
              <span className="font-medium text-gray-700">이름</span>
              <span className="ml-2 text-lg">{userInfo.userName}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">아이디(휴대폰번호)</span>
              <span className="ml-2 text-lg">{userInfo.userPhone}</span>
            </div>						
            <div>
              <span className="font-medium text-gray-700">가입날짜</span>
              <span className="ml-2 text-lg">{userInfo.joinedAt}</span>
            </div>
          </div>

					<div className="flex justify-end items-center mt-6 gap-2">
						<button
							onClick={() => navigate(`/admin/userInfo/${userPhone}/edit`)}
							className="btn btn-accent"
						>
							회원정보수정
						</button>
						<button
							onClick={handleGoBack}
							className="btn btn-secondary"
						>
							회원삭제
						</button>
					</div>

        </div>

        {/* 농장 추가 버튼 */}
        <div className="flex justify-end mb-4">
          <button className="btn btn-primary">
            농장 추가
          </button>
        </div>

        {/* 농장 목록 테이블 */}
        <div className="table-wrap">
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th>번호</th>
                  <th>농장 구분</th>
                  <th>농장 이름</th>
                  <th>농장 주소</th>
                  <th>농장 번호</th>
                  <th>농장 번호</th>
                  <th>수정</th>
                </tr>
              </thead>
              <tbody>
                {userDetailList.map((farm, index) => (
                  <tr key={farm.farmIdx}>
                    <td>{index + 1}</td>
                    <td>{farm.farmCrops}</td>
                    <td>{farm.farmName}</td>
                    <td>{farm.farmAddr}</td>
                    <td>{farm.farmPhone}</td>
                    <td>{farm.farmPhone}</td>
                    <td>
                      <div className="flex gap-1">
                        <button className="btn btn-sm btn-accent">수정</button>
                        <button className="btn btn-sm btn-secondary">삭제</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}