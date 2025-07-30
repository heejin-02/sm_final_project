// src/pages/AdminMain.jsx
// 관리자 메인 페이지 - 전체 회원 정보
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAllUsers } from '../api/admin';
import Loader from '../components/Loader';
import AddUserModal from '../components/AddUserModal';

export default function AdminMain() {
  const navigate = useNavigate();
  const [allUserList, setAllUserList] = useState([]); // 전체 데이터
  const [filteredUserList, setFilteredUserList] = useState([]); // 검색 필터링된 데이터
  const [displayedUserList, setDisplayedUserList] = useState([]); // 현재 페이지에 표시할 데이터
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchField, setSearchField] = useState('user_name');
  const [keyword, setKeyword] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10); // 페이지당 표시할 개수
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [isSearched, setIsSearched] = useState(false); // 검색 실행 여부
  const [searchedKeyword, setSearchedKeyword] = useState(''); // 실제 검색된 키워드
  const [searchedField, setSearchedField] = useState(''); // 실제 검색된 필드
  const [showAddUserModal, setShowAddUserModal] = useState(false);

  // 회원 등록 성공 시 리스트 새로고침
  const handleAddUserSuccess = () => {
    fetchAllUsers();
  };

  // 전체 회원 정보 가져오기
  const fetchAllUsers = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await getAllUsers();
      const rawUserList = response.data || [];

      // console.log('🔍 백엔드에서 받은 원본 데이터:', rawUserList.length, '건');

      // admin 계정 제외
      const filteredList = rawUserList.filter(user => user.userPhone !== 'admin');

      // console.log('🔍 admin 제외 후:', filteredList.length, '건');

      // 회원별로 중복 제거 (userPhone 기준으로 첫 번째 농장만 대표로 사용)
      const uniqueUserMap = new Map();
      filteredList.forEach(user => {
        if (!uniqueUserMap.has(user.userPhone)) {
          uniqueUserMap.set(user.userPhone, user);
        }
      });

      const userList = Array.from(uniqueUserMap.values())
        .sort((a, b) => new Date(b.joinedAt) - new Date(a.joinedAt)); // 가입일자 최신순

      // console.log('중복 제거 후 최종 회원 수:', userList.length, '명');

      setAllUserList(userList);
      setFilteredUserList(userList);
      setTotalCount(userList.length);

      const duplicatePhones = userList
        .map(user => user.userPhone)
        .filter((v, i, arr) => arr.indexOf(v) !== i);
      // if (duplicatePhones.length) {
      //   console.warn('중복된 userPhone 있음:', duplicatePhones);
      // }

      // 첫 페이지 데이터 설정
      updateDisplayedData(userList, 1);

    } catch (error) {
      console.error('회원 정보 조회 실패:', error);
      setError('데이터 요청이 실패했습니다. 서버 연결을 확인해주세요.');
      setAllUserList([]);
      setFilteredUserList([]);
      setDisplayedUserList([]);
      setTotalCount(0);
      setCurrentPage(1);
      setTotalPages(1);
    } finally {
      setLoading(false);
    }
  };

  // 표시할 데이터 업데이트 (페이징 처리)
  const updateDisplayedData = (dataList, page) => {
    const startIndex = (page - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const pageData = dataList.slice(startIndex, endIndex);

    // console.log('페이지 데이터 업데이트 - 페이지:', page);
    // console.log('전체 데이터:', dataList.length, '건');
    // console.log('표시할 데이터:', pageData.length, '건');
    // console.log('표시할 회원들:', pageData.map(u => u.userName));

    setDisplayedUserList(pageData);
    setCurrentPage(page);
    setTotalPages(Math.ceil(dataList.length / pageSize));
  };

  // 검색 처리
  const handleSearch = (e) => {
    e.preventDefault();

    // console.log('검색 시작 - 검색어:', keyword, '검색 필드:', searchField);
    // console.log('전체 데이터 수:', allUserList.length);

    if (!keyword.trim()) {
      // 검색어가 없으면 전체 데이터 표시
      setFilteredUserList(allUserList);
      updateDisplayedData(allUserList, 1);
      setIsSearched(false);
      return;
    }

    // 검색 필터링 (admin 계정은 이미 제외된 상태)
    const filtered = allUserList.filter(user => {
      const searchValue = keyword.toLowerCase();

      switch (searchField) {
        case 'user_name':
          const matches = user.userName?.toLowerCase().includes(searchValue);
          // if (matches) {
          //   console.log('매칭된 회원:', user.userName, user.userPhone);
          // }
          return matches;
        case 'farm_name':
          const farmMatches = user.farmName?.toLowerCase().includes(searchValue);
          // if (farmMatches) {
          //   console.log('매칭된 농장:', user.farmName, user.userName);
          // }
          return farmMatches;
        default:
          return false;
      }
    });

    // console.log('검색 결과:', filtered.length, '건');
    // console.log('검색된 회원들:', filtered.map(u => u.userName));

    setFilteredUserList(filtered);
    setTotalCount(filtered.length);
    updateDisplayedData(filtered, 1);

    // 검색 상태 업데이트
    setIsSearched(true);
    setSearchedKeyword(keyword);
    setSearchedField(searchField);
  };

  // 검색 초기화 (전체보기)
  const handleReset = () => {
    setKeyword('');
    setFilteredUserList(allUserList);
    setTotalCount(allUserList.length);
    updateDisplayedData(allUserList, 1);
    setIsSearched(false);
    setSearchedKeyword('');
    setSearchedField('');
  };

  // 페이지 변경
  const handlePageChange = (page) => {
    updateDisplayedData(filteredUserList, page);
  };

  // 회원 수정 페이지로 이동
  const handleEditUser = (userPhone) => {
    navigate(`/admin/userInfo/${userPhone}`);
  };

  useEffect(() => {
    fetchAllUsers();
  }, []);

  if (loading) {
    return (
      <div className="section flex items-center justify-center">
        <Loader message="회원 정보를 불러오는 중..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="section p-6">
        <div className="inner">
          <h1 className="tit-head">전체 회원 정보</h1>
          <div className="flex items-center justify-between mb-6">
            <p className="text-gray-600 mt-1">
              총 {totalCount}명의 회원 (페이지당 {pageSize}개씩 표시)
            </p>
            <button 
              onClick={() => setShowAddUserModal(true)}
              className="btn btn-primary"
            >
              회원 등록
            </button>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <div className="text-red-600 text-lg font-medium mb-2">
              ⚠️ 오류 발생
            </div>
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={() => fetchAllUsers()}
              className="btn btn-accent"
            >
              다시 시도
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="inner">
        <h1 className="tit-head">전체 회원 정보</h1>
        
        <div className="flex justify-between items-center mb-4">
          <p className="text-gray-600 mt-1">
            총 {totalCount}명의 회원 (페이지당 {pageSize}개씩 표시)
          </p>
          <button 
            onClick={() => setShowAddUserModal(true)}
            className="btn btn-primary"
          >
            회원 등록
          </button>
        </div>

        {/* 검색 폼 */}
        <div className="search-bar">
          <form onSubmit={handleSearch} className="flex gap-2 justify-between w-full">
            <div className="flex-1">
              {/* <label className="block text-sm font-medium text-gray-700 mb-2">
                검색 조건
              </label> */}
              <select
                value={searchField}
                onChange={(e) => setSearchField(e.target.value)}
                className="input"
              >
                <option value="" disabled>검색 조건</option>
                <option value="user_name">회원 이름</option>
                <option value="farm_name">농장 이름</option>
              </select>
            </div>
            <div className="flex-2">
              {/* <label className="block text-sm font-medium text-gray-700 mb-2">
                검색어
              </label> */}
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
            <button
              type="button"
              onClick={handleReset}
              className="btn btn-secondary"
            >
              검색 초기화
            </button>
          </form>
        </div>

        {/* 검색 결과 표시 */}
        {isSearched && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="text-blue-800">
                <span className="font-medium">
                  {searchedField === 'user_name' ? '회원 이름' : '농장 이름'}에서
                  "{searchedKeyword}" 검색 결과
                </span>
                <span className="ml-2 text-blue-600">
                  ({totalCount}건)
                </span>
              </div>
              <button
                onClick={handleReset}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium cursor-pointer"
              >
                ✕ 검색 해제
              </button>
            </div>
          </div>
        )}

        {/* 회원 목록 테이블 */}
        <div className="table-wrap">
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th>번호</th>
                  <th>이름</th>
                  <th>아이디(휴대폰번호)</th>
                  <th>대표농장이름 / 하우스</th>
                  <th>대표농장주소</th>
                  <th>농장번호</th>
                  <th>가입날짜</th>
                </tr>
              </thead>
              <tbody>
                {displayedUserList.length > 0 ? (
                  displayedUserList.map((user, index) => (
                    <tr
                      key={`${user.userPhone}-${user.farmIdx || index}`}
                      className="clickable"
                      onClick={() => handleEditUser(user.userPhone)}
                      data-farm-idx={user.farmIdx}
                    >
                      <td>{(currentPage - 1) * pageSize + index + 1}</td>
                      <td>
                        <span className="text-blue-600">
                          {user.userName}
                        </span>
                      </td>
                      <td>{user.userPhone}</td>
                      <td>{user.farmName}</td>
                      <td>{user.farmAddr}</td>
                      <td>{user.farmPhone}</td>
                      <td>{user.joinedAt}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="text-center text-gray-500">
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
          <div className="pg_wrap">
            <nav className="flex space-x-2">
              {/* 처음 버튼 (5페이지 이상일 때만) */}
              {totalPages >= 5 && currentPage > 1 && (
                <button className='pg-btn' onClick={() => handlePageChange(1)}>처음</button>
              )}

              {/* 이전 5페이지 그룹 버튼 */}
              {Math.ceil(currentPage / 5) > 1 && (
                <button
                  className='pg-btn'
                  onClick={() => handlePageChange(Math.max(1, Math.floor((currentPage - 1) / 5) * 5))}
                >
                  이전
                </button>
              )}

              {/* 페이지 번호들 (5개씩 그룹) */}
              {(() => {
                const currentGroup = Math.ceil(currentPage / 5);
                const startPage = (currentGroup - 1) * 5 + 1;
                const endPage = Math.min(startPage + 4, totalPages);

                return Array.from({ length: endPage - startPage + 1 }, (_, i) => startPage + i).map((page) => (
                  <button
                    key={page}
                    onClick={() => handlePageChange(page)}
                    className={`pg-btn ${page === currentPage ? 'current' : ''}`}
                  >
                    {page}
                  </button>
                ));
              })()}

              {/* 다음 5페이지 그룹 버튼 */}
              {Math.ceil(currentPage / 5) < Math.ceil(totalPages / 5) && (
                <button
                  className='pg-btn'
                  onClick={() => handlePageChange(Math.min(totalPages, Math.ceil(currentPage / 5) * 5 + 1))}
                >
                  다음
                </button>
              )}

              {/* 맨끝 버튼 (5페이지 이상일 때만) */}
              {totalPages >= 5 && currentPage < totalPages && (
                <button className='pg-btn' onClick={() => handlePageChange(totalPages)}>맨끝</button>
              )}
            </nav>
          </div>
        )}
        <AddUserModal
          isOpen={showAddUserModal}
          onClose={() => setShowAddUserModal(false)}
          onSuccess={handleAddUserSuccess}
        />
      </div>
    </div>
  );
}
