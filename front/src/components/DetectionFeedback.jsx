// src/components/DetectionFeedback.jsx
import { useState, useEffect } from 'react';
import FeedbackTextarea from './FeedbackTextarea';

// 로컬스토리지 키 설정 함수
const FEEDBACK_KEY = (anlsIdx) => `feedback-${anlsIdx}`;

// 저장된 피드백 불러오기
const getStoredFeedback = (anlsIdx) => {
  return localStorage.getItem(FEEDBACK_KEY(anlsIdx)); // 'correct' | 'wrong' | null
};

// 피드백 저장
const storeFeedback = (anlsIdx, accuracy) => {
  localStorage.setItem(FEEDBACK_KEY(anlsIdx), accuracy);
};

export default function DetectionFeedback({ anlsIdx, alertDetail, onFeedbackSubmit }) {
  const [feedback, setFeedback] = useState({ accuracy: null });
  const [isFirstSubmit, setIsFirstSubmit] = useState(false);

  // ✅ 해충 설명 메시지
  const getDefaultMessage = (insectName) => {
    const messages = {
      '꽃노랑총채벌레': '꽃노랑총채벌레는 작고 노란색의 해충으로...',
      '담배가루이': '담배가루이는 작고 흰 가루 같아 보여요...',
      '비단노린재': '비단노린재는 잎이나 열매의 수액을 빨아먹는...',
      '알락수염노린재': '알락수염노린재는 식물의 즙을 빨아먹는 해충이에요...'
    };
    return messages[insectName] || '해충이 탐지되었습니다. 분석 결과를 확인해주세요.';
  };

  // ✅ 초기 로딩 시 저장된 피드백 불러오기
  useEffect(() => {
    const saved = getStoredFeedback(anlsIdx);
    if (saved) {
      setFeedback({ accuracy: saved });
    }
  }, [anlsIdx]);

  // ✅ 버튼 클릭 핸들러
  const handleAccuracySelect = (accuracy) => {
    setFeedback({ accuracy });
    storeFeedback(anlsIdx, accuracy);
    onFeedbackSubmit?.({ accuracy });
    setIsFirstSubmit(true);
  };

  return (
    <div className="detection-feedback">

      {/* 💬 백구의 메시지 */}
      <div className="baekgu-msg-wrap">
        <div className="thumb">
          <img src="/images/talk_109.png" alt="백구" />
        </div>
        <div className="baekgu-msg">
          {alertDetail?.gptResult?.gptContent || getDefaultMessage(alertDetail?.greenhouseInfo?.insectName)}
        </div>
      </div>

      {/* 사용자 메모 입력란 */}
      <FeedbackTextarea anlsIdx={anlsIdx} alertDetail={alertDetail} />     

      {/* ✅ 피드백 질문 */}
      <div className="feedback-step">
        <div className="feedback-header">
          <h3 className="tit-2">
            {feedback.accuracy
              ? '해충 탐지 피드백'
              : '해충 탐지 결과가 정확한가요?'}            
          </h3>
        </div>
        <p className="question-text">
          {feedback.accuracy
            ? (
              <>
                <strong>{alertDetail?.greenhouseInfo?.insectName}</strong>을(를) <strong>{alertDetail?.greenhouseInfo?.ghName}</strong>에서 탐지함에 대해
              </>
            )
            : (
              <>
                <strong>{alertDetail?.greenhouseInfo?.insectName}</strong>을(를) <strong>{alertDetail?.greenhouseInfo?.ghName}</strong>에서 탐지했다고 하는데, 맞나요?
              </>
            )
          }
        </p>

        <div className="feedback-options-simple">
          <button
            className={`btn btn-lg feedback-btn-simple correct ${feedback.accuracy === 'correct' ? 'selected' : ''}`}
            onClick={() => handleAccuracySelect('correct')}
            disabled={!!feedback.accuracy}
          >
            <span className="btn-icon">✅</span>
            <span className="btn-text">정확함</span>
          </button>

          <button
            className={`btn btn-lg feedback-btn-simple wrong ${feedback.accuracy === 'wrong' ? 'selected' : ''}`}
            onClick={() => handleAccuracySelect('wrong')}
            disabled={!!feedback.accuracy}
          >
            <span className="btn-icon">❌</span>
            <span className="btn-text">틀림</span>
          </button>
        </div>

        {/* ✅ 피드백 완료 메시지는 딱 1회만 */}
        {isFirstSubmit && (
          <p className="complete-message text-green-600 mt-4 text-center">
            ✅ 피드백이 완료되었습니다. 감사합니다!
          </p>
        )}
      </div>
    </div>
  );
}
