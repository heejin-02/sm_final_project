// src/components/DetectionFeedback.jsx
import { useState } from 'react';

export default function DetectionFeedback({ notification, onFeedbackSubmit, onMarkAsRead }) {
  const [step, setStep] = useState(1);
  const [feedback, setFeedback] = useState({
    accuracy: null,
    wrongType: null,
    wrongReason: null,
    correctRegion: null,
    countIssue: null,
    environment: [],
    timeOfDay: null,
    improvement: []
  });

  // 기본 정확도 평가 - 바로 완료 처리
  const handleAccuracySelect = (accuracy) => {
    setFeedback(prev => ({ ...prev, accuracy }));
    setStep(3);
    handleSubmit({ ...feedback, accuracy });
  };



  // 피드백 제출
  const handleSubmit = (finalFeedback = feedback) => {
    console.log('피드백 제출:', finalFeedback);
    onFeedbackSubmit?.(finalFeedback);
    setStep(3);
  };

  // 나중에 확인하기
  const handleMarkAsRead = () => {
    onMarkAsRead?.(notification.id);
  };



  return (

    <div className="detection-feedback">

      {/* 백구의 탐지 분석 결과 텍스트 */}
      <div className="baekgu-msg-wrap">
        <div className="thumb">
          <img src="/images/talk_109.png" alt="" />
        </div>
        <div className="baekgu-msg">
          해당 벌레는 86% 확률로 ‘진딧물’로 의심돼요! 비슷한 위치에서 어제 오후 3시, 오늘 오전 5시에 감지된 것과 비슷해요. 알려드린 위치에 진딧물이 번식하고 있을 확률이 높아요!
        </div>
      </div>       

      {step === 1 && (
       
        <div className="feedback-step">
          <div className="feedback-question">

            <div className="feedback-header">
              <h3 className="tit-2">해충 탐지 결과가 정확한가요?</h3>
              {/* <p className="feedback-subtitle">
                사장님의 답변으로 백구가 더 똑똑해집니다!
              </p> */}
            </div>

            <p className="question-text">
              <strong>{notification.bugName}</strong>을(를) <strong>{notification.location}</strong>에서 탐지했다고 하는데, 맞나요?
            </p>            

          </div>
          
          <div className="feedback-options-simple">
            <button
              className="btn btn-lg feedback-btn-simple correct"
              onClick={() => handleAccuracySelect('correct')}
            >
              <span className="btn-icon">✅</span>
              <span className="btn-text">정확함</span>
            </button>

            <button
              className="btn btn-lg feedback-btn-simple wrong"
              onClick={() => handleAccuracySelect('wrong')}
            >
              <span className="btn-icon">❌</span>
              <span className="btn-text">틀림</span>
            </button>

            <button
              className="btn btn-lg feedback-btn-simple later"
              onClick={handleMarkAsRead}
            >
              <span className="btn-icon">📋</span>
              <span className="btn-text">나중에 확인하기</span>
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="feedback-complete">
          <div className="complete-icon">✅</div>
          <h3 className="complete-title">피드백이 완료되었습니다!</h3>
          <p className="complete-message">
            소중한 의견 감사합니다. 백구가 더 똑똑해질 수 있도록 도와주셨어요!
          </p>
        </div>
      )}
    </div>
  );
}
