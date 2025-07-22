// Detail.jsx
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; // react-router-dom v6 기준
import { diaryData, saveDiaryData } from '../diary-data';

export default function Detail() {
  const location = useLocation();
  const navigate = useNavigate();
  const [diary, setDiary] = useState(null);

  // 1) 해시 대신 react-router 파라미터 예시
  //    URL 예: /detail?id=123
  useEffect(() => {
    const query = new URLSearchParams(location.search);
    const id = query.get('id');
    const found = diaryData.find((d) => String(d.id) === id) || null;
    setDiary(found);
  }, [location.search]);

  // 2) 삭제 핸들러
  const handleDelete = () => {
    if (!diary) return;
    if (!window.confirm('이 일기를 정말 삭제하시겠습니까?')) return;

    const idx = diaryData.findIndex((d) => d.id === diary.id);
    if (idx !== -1) {
      diaryData.splice(idx, 1);
      saveDiaryData();
      alert('일기가 삭제되었습니다.');
      navigate('/'); // 홈으로 리다이렉트
    }
  };

  // 3) 로딩 또는 없는 경우 처리
  if (diary === undefined) {
    // 아직 검색 중
    return <p>불러오는 중...</p>;
  }
  if (diary === null) {
    return <p className="text-gray-500">일기를 찾을 수 없습니다.</p>;
  }

  // 4) 실제 렌더링
  return (
    <main className="p-4">
      <ul>
        <li
          className="diary-item bg-gray-200 p-4 rounded-lg mb-5"
          data-id={diary.id}
        >
          <p className="mb-5">{diary.content}</p>
          <footer className="flex justify-between items-center">
            <time
              className="text-sm text-gray-500"
              dateTime={diary.date}
            >
              {diary.date}
            </time>
            <button
              onClick={handleDelete}
              className="p-1 hover:bg-red-100 rounded"
              aria-label="삭제"
            >
              {/* SVG 아이콘 */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
              >
                <path
                  fill="#dc3545"
                  d="m20.37 8.91l-1 1.73l-12.13-7l1-1.73l3.04 1.75l1.36-.37l4.33 2.5l.37 1.37zM6 19V7h5.07L18 11v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2"
                />
              </svg>
            </button>
          </footer>
        </li>
      </ul>
    </main>
  );
}
