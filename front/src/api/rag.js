// src/api/rag.js
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8095';

// RAG 챗봇에 해충 정보 질문
export const askInsectInfo = async (insectName) => {
  try {
    const response = await axios.post(`${BASE_URL}/ml/chat`, {
      insect: insectName,
      question: '이 해충의 특성과 방제 방법을 알려주세요',
    });
    return response.data;
  } catch (error) {
    console.error('RAG API 호출 실패:', error);
    throw error;
  }
};

// 일반 질문
export const askQuestion = async (question) => {
  try {
    const response = await axios.post(`${BASE_URL}/ml/ask`, {
      question,
    });
    return response.data;
  } catch (error) {
    console.error('RAG API 호출 실패:', error);
    throw error;
  }
};
