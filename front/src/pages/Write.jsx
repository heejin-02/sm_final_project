// src/components/Write.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { diaryData, saveDiaryData } from '../diary-data';

export default function Write() {
  const navigate = useNavigate();
  const today = new Date().toISOString().split('T')[0]; // "YYYY-MM-DD"
  const [content, setContent] = useState('');
  const [date] = useState(today);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!content.trim()) {
      alert('일기를 작성해주세요.');
      return;
    }

    const newDiary = {
      id: crypto.randomUUID(),
      content,
      date,
    };

    diaryData.push(newDiary);
    saveDiaryData(); // localStorage에 저장
    console.log('New diary added:', newDiary);

    navigate('/'); // 홈으로 이동
  };

  return (
    <main className="p-4">
      <form
        className="diary-form bg-gray-200 p-4 rounded-lg mb-5 flex flex-col"
        onSubmit={handleSubmit}
      >
        <textarea
          id="diary-content"
          className="w-full h-52 p-2 rounded-lg mb-3 border"
          placeholder="일기를 작성하세요.."
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        <input
          type="text"
          id="diary-date"
          className="text-gray-400 text-sm mb-4"
          value={date}
          disabled
        />
        <button
          type="submit"
          className="bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
        >
          저장
        </button>
      </form>
    </main>
  );
}
