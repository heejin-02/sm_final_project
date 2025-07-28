// src/pages/AdminMain.jsx
// 관리자 메인 페이지 - 전체 회원 정보
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Loader from '../components/Loader';

export default function AdminMain() {
  const navigate = useNavigate();
  const [farmList, setFarmList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchField, setSearchField] = useState('user_name');
  const [keyword, setKeyword] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);

  // 회원 정보 가져오기
  const fetchFarmList = async (page = 1, search = '') => {
    try {
      setLoading(true);
      // TODO: 실제 API 엔드포인트로 변경
      const response = await fetch(`/api/admin/farms?page=${page}&searchField=${searchField}&keyword=${search}`);
      const data = await response.json();

      setFarmList(data.farmList || []);
      setCurrentPage(data.currentPage || 1);
      setTotalPages(data.totalPages || 1);
      setTotalCount(data.totalCount || 0);
    } catch (error) {
      console.error('회원 정보 조회 실패:', error);
      // 임시 더미 데이터
      setFarmList([
        {
          farmIdx: 1,
          userName: '김농부',
          userPhone: '010-1234-5678',
          farmName: '김농부네 농장',
          farmAddr: '서울시 강남구',
          farmPhone: '02-1234-5678',
          joinedAt: '2024-01-15'
        },
        {
          farmIdx: 2,
          userName: '이농부',
          userPhone: '010-2345-6789',
          farmName: '이농부네 농장',
          farmAddr: '경기도 수원시',
          farmPhone: '031-2345-6789',
          joinedAt: '2024-02-20'
        }
      ]);
      setTotalCount(2);
    } finally {
      setLoading(false);
    }
  };

  // 검색 처리
  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1);
    fetchFarmList(1, keyword);
  };

  // 페이지 변경
  const handlePageChange = (page) => {
    setCurrentPage(page);
    fetchFarmList(page, keyword);
  };

  // 회원 추가 페이지로 이동
  const handleAddUser = () => {
    navigate('/admin/add-user');
  };

  // 회원 수정 페이지로 이동
  const handleEditUser = (userPhone) => {
    navigate(`/admin/edit-user/${userPhone}`);
  };

  useEffect(() => {
    fetchFarmList();
  }, []);

  if (loading) {
    return (
      <div className="section flex items-center justify-center">
        <Loader message="회원 정보를 불러오는 중..." />
      </div>
    );
  }

  return (
    <div className="section p-6">
      <div className="max-w-7xl mx-auto">
        {/* 헤더 */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">전체 회원 정보</h1>
            <p className="text-gray-600 mt-1">총 {totalCount}명의 회원이 등록되어 있습니다.</p>
          </div>
          <button
            onClick={handleAddUser}
            className="btn btn-primary"
          >
            회원 추가
          </button>
        </div>

        {/* 검색 폼 */}
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <form onSubmit={handleSearch} className="flex gap-4 items-end">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                검색 조건
              </label>
              <select
                value={searchField}
                onChange={(e) => setSearchField(e.target.value)}
                className="input"
              >
                <option value="user_name">회원 이름</option>
                <option value="farm_name">농장 이름</option>
                <option value="farm_addr">농장 주소</option>
              </select>
            </div>
            <div className="flex-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                검색어
              </label>
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="검색어를 입력하세요"
                className="input"
              />
            </div>
            <button type="submit" className="btn btn-accent">
              검색
            </button>
          </form>
        </div>

        {/* 회원 목록 테이블 */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    번호
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    이름
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    아이디(휴대폰번호)
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    대표농장이름
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    대표농장주소
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    농장번호
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    가입날짜
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {farmList.length > 0 ? (
                  farmList.map((farm) => (
                    <tr key={farm.farmIdx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {farm.farmIdx}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          onClick={() => handleEditUser(farm.userPhone)}
                          className="text-sm font-medium text-blue-600 hover:text-blue-900"
                        >
                          {farm.userName}
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {farm.userPhone}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {farm.farmName}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {farm.farmAddr}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {farm.farmPhone}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {farm.joinedAt}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="px-6 py-4 text-center text-sm text-gray-500">
                      검색 결과가 없습니다.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* 페이지네이션 */}
        {totalPages > 1 && (
          <div className="flex justify-center mt-6">
            <nav className="flex space-x-2">
              {currentPage > 1 && (
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  이전
                </button>
              )}

              {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                <button
                  key={page}
                  onClick={() => handlePageChange(page)}
                  className={`px-3 py-2 text-sm font-medium rounded-md ${
                    page === currentPage
                      ? 'text-white bg-blue-600 border border-blue-600'
                      : 'text-gray-500 bg-white border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {page}
                </button>
              ))}

              {currentPage < totalPages && (
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  다음
                </button>
              )}
            </nav>
          </div>
        )}
      </div>
    </div>
  );
}
