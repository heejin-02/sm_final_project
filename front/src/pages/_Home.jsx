// src/components/Home.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { diaryData, saveDiaryData } from '../diary-data';

export default function Home() {
  const navigate = useNavigate();
  const [diaries, setDiaries] = useState([]);

  // 1) 컴포넌트가 마운트될 때 diaryData 정렬하여 state에 저장
  useEffect(() => {
    const sorted = [...diaryData].sort(
      (a, b) => new Date(b.date) - new Date(a.date)
    );
    setDiaries(sorted);
  }, []);

  // 2) 삭제 핸들러
  const handleDelete = (id) => {
    if (!window.confirm('정말 삭제하시겠습니까?')) return;

    const idx = diaryData.findIndex((d) => d.id === id);
    if (idx !== -1) {
      diaryData.splice(idx, 1);
      saveDiaryData();
      // state에서 제거
      setDiaries((prev) => prev.filter((d) => d.id !== id));
    }
  };

  // 3) 렌더링
  return (
    <main className="p-4">
      <ul>
        {diaries.map((diary) => (
          <li
            key={diary.id}
            className="diary-item"
            onClick={() => navigate(`/detail?id=${diary.id}`)}
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
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(diary.id);
                }}
                className="delete-button"
                aria-label="삭제"
              >
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
        ))}
      </ul>

      {/* 4) 글 추가 버튼 */}
      <button
        className="add-button"
        onClick={() => navigate('/write')}
        aria-label="글 추가"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="64"
          height="64"
          viewBox="0 0 24 24"
        >
          <path
            fill="#007aff"
            d="M17 13h-4v4h-2v-4H7v-2h4V7h2v4h4m-5-9A10 10 0 0 0 2 12a10 10 0 0 0 10 10a10 10 0 0 0 10-10A10 10 0 0 0 12 2"
          />
        </svg>
      </button>
    </main>
  );
}
