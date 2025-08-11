// src/pages/AdminFarmInfo.jsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { API } from '../api/http';
import { getGreenhousesByFarm, saveGreenhouses } from '../api/greenhouse';

function AdminFarmInfo() {
  const { farmIdx } = useParams();            // string
  const navigate = useNavigate();
  const location = useLocation();

  // 추가/수정 모드 구분
  const { userInfo: passedUserInfo, farmInfo: passedFarmInfo, mode } = location.state || {};
  const isCreateMode = farmIdx === 'create' || mode === 'create';

  const [farmInfo, setFarmInfo] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  // 수정용 상태
  const [editedFarmInfo, setEditedFarmInfo] = useState({
    farmName: '',
    farmAddr: '',
    farmPhone: '',
    farmCrops: '',
    farmArea: '',
    farmImg: ''
  });

  // 파일 업로드(추가 모드)
  const [selectedFile, setSelectedFile] = useState(null);

  // 구역 관리 상태
  const [regions, setRegions] = useState([]);

  // 기본 9개 구역 생성 유틸
  const makeDefaultRegions = (fid = null) => {
    const arr = [];
    for (let i = 1; i <= 9; i++) {
      arr.push({
        id: i,
        name: '',
        ghIdx: i,
        farmIdx: fid === null ? null : Number(fid),
        ghArea: '',
        ghCrops: ''
      });
    }
    return arr;
  };

  // 구역 정보 로드 (API 기반)
  const loadRegionData = async (fid) => {
    if (!fid) return;
    try {
      const apiRegions = await getGreenhousesByFarm(fid);
      if (apiRegions && apiRegions.length > 0) {
        const regionsForAdmin = apiRegions.map((region) => ({
          id: region.ghIdx,
          name: region.ghName || '',
          ghIdx: region.ghIdx,
          farmIdx: region.farmIdx,
          ghArea: region.ghArea || '',
          ghCrops: region.ghCrops || ''
        }));
        setRegions(regionsForAdmin);
        // console.log('API에서 구역 정보 로드 완료:', regionsForAdmin);
      } else {
        throw new Error('No greenhouse data found');
      }
    } catch (err) {
      console.error('구역 정보 로드 실패:', err);
      setRegions(makeDefaultRegions(fid));
      // console.log('기본 구역 정보 로드 완료');
    }
  };

  // 구역 이름 변경
  const updateRegionName = (id, name) => {
    setRegions((prev) =>
      prev.map((region) => (region.id === id ? { ...region, name } : region))
    );
  };

  // 구역 저장/수정
  const handleRegionSubmit = async () => {
    try {
      if (!farmInfo?.farmIdx && !isCreateMode) {
        alert('농장 정보가 없습니다.');
        return;
      }
      const targetFarmIdx = farmInfo?.farmIdx || Number(farmIdx);

      const regionData = regions.map((region) => ({
        farmIdx: targetFarmIdx,
        ghIdx: region.ghIdx || region.id,
        ghName: region.name || `${region.ghIdx || region.id}번 구역`,
        ghArea: region.ghArea || '100m²',
        ghCrops: region.ghCrops || '토마토',
        createdAt: new Date().toISOString()
      }));

      // console.log('구역 데이터 저장 시도:', regionData);
      await saveGreenhouses(targetFarmIdx, regionData);
      alert(`${regions.length}개 구역이 저장되었습니다.`);
    } catch (err) {
      console.error('구역 저장 실패:', err);
      if (err.response?.status === 404) {
        alert('구역 저장 API가 아직 구현되지 않았습니다.\n백엔드 개발 완료 후 사용 가능합니다.');
      } else {
        alert('구역 저장에 실패했습니다.\n잠시 후 다시 시도해주세요.');
      }
    }
  };

  // 농장 정보 로드
  useEffect(() => {
    const fetchFarmInfo = async () => {
      try {
        setLoading(true);

        if (isCreateMode) {
          // 추가 모드
          if (passedUserInfo) setUserInfo(passedUserInfo);
          setFarmInfo(null);
          setIsEditing(true); // 추가 모드에서 바로 편집
          setRegions(makeDefaultRegions(null));
        } else {
          // 수정 모드
          if (passedFarmInfo && passedUserInfo) {
            // 전달받은 데이터 사용
            setFarmInfo(passedFarmInfo);
            setUserInfo(passedUserInfo);
            setEditedFarmInfo({
              farmName: passedFarmInfo.farmName || '',
              farmAddr: passedFarmInfo.farmAddr || '',
              farmPhone: passedFarmInfo.farmPhone || '',
              farmCrops: passedFarmInfo.farmCrops || '',
              farmArea: passedFarmInfo.farmArea || '',
              farmImg: passedFarmInfo.farmImg || ''
            });
            await loadRegionData(passedFarmInfo.farmIdx);
          } else {
            // API에서 조회
            const res = await API.get(`/api/farms/${farmIdx}/detail`);
            setFarmInfo(res.data);
            setEditedFarmInfo({
              farmName: res.data.farmName || '',
              farmAddr: res.data.farmAddr || '',
              farmPhone: res.data.farmPhone || '',
              farmCrops: res.data.farmCrops || '',
              farmArea: res.data.farmArea || '',
              farmImg: res.data.farmImg || ''
            });
            await loadRegionData(farmIdx);
          }
        }
      } catch (err) {
        console.error('농장 정보 조회 실패:', err);
        setError('농장 정보를 불러오는데 실패했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchFarmInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [farmIdx, location.state, isCreateMode]);

  // 수정 시작/취소/저장
  const handleStartEdit = () => setIsEditing(true);

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedFarmInfo({
      farmName: farmInfo?.farmName || '',
      farmAddr: farmInfo?.farmAddr || '',
      farmPhone: farmInfo?.farmPhone || '',
      farmCrops: farmInfo?.farmCrops || '',
      farmArea: farmInfo?.farmArea || '',
      farmImg: farmInfo?.farmImg || ''
    });
  };

  const handleSaveEdit = async () => {
    // 유효성 검사
    if (!editedFarmInfo.farmName.trim()) {
      alert('농장 이름을 입력해주세요.');
      return;
    }
    if (!editedFarmInfo.farmAddr.trim()) {
      alert('농장 주소를 입력해주세요.');
      return;
    }

    try {
      if (isCreateMode) {
        // 농장 추가
        if (!passedUserInfo?.userPhone) {
          alert('농장주 정보가 없습니다.');
          return;
        }
        const formData = new FormData();
        formData.append('farmName', editedFarmInfo.farmName);
        formData.append('farmAddr', editedFarmInfo.farmAddr);
        formData.append('farmPhone', editedFarmInfo.farmPhone);
        formData.append('farmCrops', editedFarmInfo.farmCrops);
        formData.append('farmArea', editedFarmInfo.farmArea);
        formData.append('userPhone', passedUserInfo.userPhone);
        if (selectedFile) formData.append('farmImg', selectedFile);

        const response = await API.post('/api/farm/insertFarm', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        if (response.status === 200) {
          alert('농장이 추가되었습니다.');
          navigate(-1);
        }
      } else {
        // 농장 수정
        const response = await API.put(`/api/admin/users/farms/${farmIdx}`, editedFarmInfo);
        if (response.status === 200) {
          setFarmInfo({ ...farmInfo, ...editedFarmInfo });
          setIsEditing(false);
          alert('농장 정보가 수정되었습니다.');
        }
      }
    } catch (err) {
      console.error('농장 정보 저장 실패:', err);
      alert(isCreateMode ? '농장 추가에 실패했습니다.' : '수정에 실패했습니다.');
    }
  };

  // 농장 삭제
  const handleDeleteFarm = async () => {
    if (
      window.confirm(
        `정말로 "${farmInfo?.farmName}" 농장을 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.`
      )
    ) {
      try {
        const response = await API.delete(`/api/admin/users/farm/${farmIdx}`);
        if (response.status === 200) {
          alert('농장이 삭제되었습니다.');
          navigate(-1);
        }
      } catch (err) {
        console.error('농장 삭제 실패:', err);
        alert('삭제에 실패했습니다.');
      }
    }
  };

  // 목록으로
  const handleGoBack = () => navigate(-1);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-lg">로딩 중...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  if (!farmInfo && !isCreateMode) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-gray-500">농장 정보를 찾을 수 없습니다.</div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="inner inner_1080">
        <h1 className="tit-head">{isCreateMode ? '농장 추가' : '농장 상세 정보'}</h1>

        {/* 상단 네비게이션 */}
        <div className="flex items-center justify-end mb-8">
          <button onClick={handleGoBack} className="btn btn-secondary">
            목록으로
          </button>
        </div>

        {/* 농장주 정보 카드 (읽기 전용) */}
        {userInfo && (
          <div className="admForm">
            <h3>농장주 정보</h3>
            <div className="admForm-ul">
              <div className="input-group flex-06">
                <label>이름</label>
                <div className="input">{userInfo.userName}</div>
              </div>
              <div className="input-group flex-08">
                <label>아이디(휴대폰번호)</label>
                <div className="input">{userInfo.userPhone}</div>
              </div>
              <div className="input-group flex-1">
                <label>가입날짜</label>
                <div className="input">{userInfo.joinedAt}</div>
              </div>
            </div>
          </div>
        )}

        {/* 농장 기본 정보 카드 */}
        <div className="admForm">
          <div className="flex justify-between items-start mb-6">
            <h3>{isCreateMode ? '농장 정보 입력' : '농장 기본 정보'}</h3>
            <div className="flex gap-2">
              {isCreateMode ? (
                <>
                  <button onClick={handleSaveEdit} className="btn btn-sm btn-accent">
                    저장
                  </button>
                  <button onClick={handleGoBack} className="btn btn-sm btn-secondary">
                    취소
                  </button>
                </>
              ) : !isEditing ? (
                <>
                  <button onClick={handleStartEdit} className="btn btn-sm btn-primary">
                    수정
                  </button>
                  <button
                    onClick={handleDeleteFarm}
                    className="btn btn-sm btn-secondary text-red-600 hover:bg-red-50"
                  >
                    삭제
                  </button>
                </>
              ) : (
                <>
                  <button onClick={handleSaveEdit} className="btn btn-sm btn-accent">
                    저장
                  </button>
                  <button onClick={handleCancelEdit} className="btn btn-sm btn-secondary">
                    취소
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="admForm-ul flex-col">
            {/* 농장 인덱스 (읽기 전용) - 추가 모드에서는 숨김 */}
            {!isCreateMode && (
              <div className="input-group">
                <label>농장 ID</label>
                <input className="input" value={farmInfo?.farmIdx || ''} readOnly />
              </div>
            )}

            {/* 농장 이름 */}
            <div className="input-group">
              <label>농장 이름</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmName || ''} readOnly />
              ) : (
                <div>
                  <input
                    type="text"
                    value={editedFarmInfo.farmName}
                    onChange={(e) =>
                      setEditedFarmInfo({ ...editedFarmInfo, farmName: e.target.value })
                    }
                    className={`input ${
                      !editedFarmInfo.farmName.trim()
                        ? 'border-red-300 focus:border-red-500'
                        : 'border-gray-300 focus:border-blue-500'
                    }`}
                    placeholder="농장 이름을 입력하세요"
                  />
                  {!editedFarmInfo.farmName.trim() && (
                    <p className="text-red-500 text-sm mt-1">농장 이름을 입력해주세요.</p>
                  )}
                </div>
              )}
            </div>

            {/* 농장 주소 */}
            <div className="input-group">
              <label>농장 주소</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmAddr || ''} readOnly />
              ) : (
                <div>
                  <input
                    type="text"
                    value={editedFarmInfo.farmAddr}
                    onChange={(e) =>
                      setEditedFarmInfo({ ...editedFarmInfo, farmAddr: e.target.value })
                    }
                    className={`input ${
                      !editedFarmInfo.farmAddr.trim()
                        ? 'border-red-300 focus:border-red-500'
                        : 'border-gray-300 focus:border-blue-500'
                    }`}
                    placeholder="농장 주소를 입력하세요"
                  />
                  {!editedFarmInfo.farmAddr.trim() && (
                    <p className="text-red-500 text-sm mt-1">농장 주소를 입력해주세요.</p>
                  )}
                </div>
              )}
            </div>

            {/* 농장 전화번호 */}
            <div className="input-group">
              <label>농장 전화번호</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmPhone || '-'} readOnly />
              ) : (
                <input
                  type="text"
                  value={editedFarmInfo.farmPhone}
                  onChange={(e) =>
                    setEditedFarmInfo({ ...editedFarmInfo, farmPhone: e.target.value })
                  }
                  className="input"
                  placeholder="농장 전화번호를 입력하세요"
                />
              )}
            </div>

            {/* 재배 작물 */}
            <div className="input-group">
              <label>재배 작물</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmCrops || '-'} readOnly />
              ) : (
                <input
                  type="text"
                  value={editedFarmInfo.farmCrops}
                  onChange={(e) =>
                    setEditedFarmInfo({ ...editedFarmInfo, farmCrops: e.target.value })
                  }
                  className="input"
                  placeholder="재배 작물을 입력하세요"
                />
              )}
            </div>

            {/* 농장 면적 */}
            <div className="input-group">
              <label>농장 면적</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmArea || '-'} readOnly />
              ) : (
                <input
                  type="text"
                  value={editedFarmInfo.farmArea}
                  onChange={(e) =>
                    setEditedFarmInfo({ ...editedFarmInfo, farmArea: e.target.value })
                  }
                  className="input"
                  placeholder="농장 면적을 입력하세요"
                />
              )}
            </div>

            {/* 농장 이미지 */}
            <div className="input-group">
              <label>농장 이미지</label>
              {!isEditing ? (
                <input className="input" value={farmInfo?.farmImg || '-'} readOnly />
              ) : isCreateMode ? (
                <div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    className="input"
                  />
                  {selectedFile && (
                    <p className="text-sm text-gray-600 mt-1">선택된 파일: {selectedFile.name}</p>
                  )}
                </div>
              ) : (
                <div className="input text-gray-500 bg-gray-100">
                  {farmInfo?.farmImg ? farmInfo.farmImg : '등록된 이미지가 없습니다'}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 구역 관리 */}
        <div className="admForm">
          <div className="admForm-header">
            <h3>구역 관리</h3>
          </div>

          <div className="admForm-ul flex-wrap flex-wrap-3">
            {regions.map((region, index) => (
              <div key={region.id} className="input-group">
                <div className="flex items-center gap-2">
                  <label className="flex-shrink-0">구역 {index + 1}</label>
                </div>
                <input
                  type="text"
                  className="input"
                  value={region.name}
                  onChange={(e) => updateRegionName(region.id, e.target.value)}
                  placeholder="구역 이름을 입력하세요"
                />
              </div>
            ))}
          </div>

          <div className="flex justify-center mt-6">
            <button onClick={handleRegionSubmit} className="btn btn-accent">
              구역 등록/수정
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminFarmInfo;
