// src/components/DetectionFeedback.jsx
import React, { useState } from 'react';

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

  // 1단계: 기본 정확도 평가
  const handleAccuracySelect = (accuracy) => {
    setFeedback(prev => ({ ...prev, accuracy }));
    
    if (accuracy === 'correct' || accuracy === 'unknown') {
      // 정확함 또는 잘 모르겠음 선택 시 바로 완료
      setStep(3);
      handleSubmit({ ...feedback, accuracy });
    } else {
      // 부분적으로 맞음 또는 틀림 선택 시 상세 피드백으로
      setStep(2);
    }
  };

  // 2단계: 상세 피드백 처리
  const handleDetailFeedback = (type, value) => {
    setFeedback(prev => {
      const newFeedback = { ...prev };
      
      if (type === 'environment' || type === 'improvement') {
        // 다중 선택 가능한 항목들
        const currentArray = newFeedback[type] || [];
        if (currentArray.includes(value)) {
          newFeedback[type] = currentArray.filter(item => item !== value);
        } else {
          newFeedback[type] = [...currentArray, value];
        }
      } else {
        newFeedback[type] = value;
      }
      
      return newFeedback;
    });
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

  // 다시 시작
  const handleRestart = () => {
    setStep(1);
    setFeedback({
      accuracy: null,
      wrongType: null,
      wrongReason: null,
      correctRegion: null,
      countIssue: null,
      environment: [],
      timeOfDay: null,
      improvement: []
    });
  };

  return (
    <div className="detection-feedback">
      <div className="feedback-header">
        <h3 className="tit-2">해충 탐지 결과가 정확한가요?</h3>
        <p className="feedback-subtitle">
          사장님의 답변으로 백구가 더 똑똑해집니다!
        </p>
      </div>

      {step === 1 && (
        <div className="feedback-step">
          <div className="feedback-question">
            <p className="question-text">
              <strong>{notification.bugName}</strong>을(를) <strong>{notification.location}</strong>에서 탐지했다고 하는데, 맞나요?
            </p>
          </div>
          
          <div className="feedback-options">
            <button 
              className="feedback-btn correct"
              onClick={() => handleAccuracySelect('correct')}
            >
              <span className="btn-icon">✅</span>
              <span className="btn-text">정확함</span>
              <span className="btn-sub">종류와 위치가 맞아요</span>
            </button>
            
            <button 
              className="feedback-btn partial"
              onClick={() => handleAccuracySelect('partial')}
            >
              <span className="btn-icon">⚠️</span>
              <span className="btn-text">부분적으로 맞음</span>
              <span className="btn-sub">일부만 맞아요</span>
            </button>
            
            <button 
              className="feedback-btn wrong"
              onClick={() => handleAccuracySelect('wrong')}
            >
              <span className="btn-icon">❌</span>
              <span className="btn-text">틀림</span>
              <span className="btn-sub">완전히 다르거나 해충이 아니에요</span>
            </button>
            
            <button
              className="feedback-btn unknown"
              onClick={() => handleAccuracySelect('unknown')}
            >
              <span className="btn-icon">❓</span>
              <span className="btn-text">잘 모르겠음</span>
              <span className="btn-sub">전문가 확인이 필요해요</span>
            </button>
          </div>

          <div className="feedback-later">
            <button
              className="feedback-later-btn"
              onClick={handleMarkAsRead}
            >
              📋 나중에 확인하기
            </button>
            <p className="later-description">
              지금은 평가하지 않고 알림만 확인 처리합니다
            </p>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="feedback-step">
          <div className="feedback-question">
            <p className="question-text">어떤 부분이 틀렸나요? (여러 개 선택 가능)</p>
          </div>


            <div className="feedback-section">
              <h4 className="section-title">🐛 해충 종류가 틀렸다면?</h4>
              <div className="feedback-grid">
                {['진딧물', '나방', '거미', '총채벌레', '응애', '해충이 아님'].map(type => (
                  <button
                    key={type}
                    className={`feedback-option ${feedback.wrongType === type ? 'selected' : ''}`}
                    onClick={() => handleDetailFeedback('wrongType', type)}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>


          <div className="feedback-section">
            <h4 className="section-title">📍 위치가 틀렸다면?</h4>
            <div className="feedback-grid">
              {['A구역', 'B구역', 'C구역', 'D구역', 'E구역', 'F구역', 'G구역', 'H구역', 'I구역'].map(region => (
                <button
                  key={region}
                  className={`feedback-option ${feedback.correctRegion === region ? 'selected' : ''}`}
                  onClick={() => handleDetailFeedback('correctRegion', region)}
                >
                  {region}
                </button>
              ))}
            </div>
          </div>
{/* 
          <div className="feedback-section">
            <h4 className="section-title">🔢 개수가 틀렸다면?</h4>
            <div className="feedback-grid">
              {['더 많음', '더 적음', '정확한 개수 모름'].map(count => (
                <button
                  key={count}
                  className={`feedback-option ${feedback.countIssue === count ? 'selected' : ''}`}
                  onClick={() => handleDetailFeedback('countIssue', count)}
                >
                  {count}
                </button>
              ))}
            </div>
          </div> */}

          <div className="feedback-actions">
            <button className="feedback-submit-btn" onClick={() => handleSubmit()}>
              피드백 제출하기
            </button>
            <button className="feedback-back-btn" onClick={() => {
              // 2단계에서 선택한 상세 피드백들만 초기화
              setFeedback(prev => ({
                ...prev,
                wrongType: null,
                wrongReason: null,
                correctRegion: null,
                countIssue: null,
                environment: [],
                timeOfDay: null,
                improvement: []
              }));
              setStep(1);
            }}>
              이전으로
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="feedback-step">
          <div className="feedback-complete">
            <div className="complete-icon">🎉</div>
            <h3 className="complete-title">평가 완료되었습니다!</h3>
            <p className="complete-message">
              소중한 의견이 백구 학습에 반영되어<br/>
              더 정확한 해충 탐지가 가능해집니다.
            </p>

            {/* 제출된 피드백 내용 표시 */}
            <div className="feedback-summary">
              <h4 className="summary-title">📝 제출된 피드백</h4>
              <div className="summary-content">
                <div className="summary-item">
                  <span className="summary-label">정확도 평가:</span>
                  <span className="summary-value">
                    {feedback.accuracy === 'correct' && '✅ 정확함'}
                    {feedback.accuracy === 'partial' && '⚠️ 부분적으로 맞음'}
                    {feedback.accuracy === 'wrong' && '❌ 틀림'}
                    {feedback.accuracy === 'unknown' && '❓ 잘 모르겠음'}
                  </span>
                </div>

                {feedback.wrongType && (
                  <div className="summary-item">
                    <span className="summary-label">올바른 해충:</span>
                    <span className="summary-value">{feedback.wrongType}</span>
                  </div>
                )}

                {feedback.correctRegion && (
                  <div className="summary-item">
                    <span className="summary-label">올바른 위치:</span>
                    <span className="summary-value">{feedback.correctRegion}</span>
                  </div>
                )}

                {feedback.countIssue && (
                  <div className="summary-item">
                    <span className="summary-label">개수 문제:</span>
                    <span className="summary-value">{feedback.countIssue}</span>
                  </div>
                )}
              </div>
            </div>

            <button className="feedback-restart-btn" onClick={handleRestart}>
              다시 평가하기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
