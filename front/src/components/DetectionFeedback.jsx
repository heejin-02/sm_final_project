// src/components/DetectionFeedback.jsx
import { useState } from 'react';

export default function DetectionFeedback({ alertDetail, onFeedbackSubmit, onMarkAsRead }) {
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

  // 해충별 기본 메시지
  const getDefaultMessage = (insectName) => {
    const messages = {
      '꽃노랑총채벌레': '꽃노랑총채벌레는 작고 노란색의 해충으로, 주로 식물의 즙을 빨아먹고 자랍니다. 이 해충이 많이 발생하면 식물이 약해지고, 결국에는 수확량이 줄어들 수 있으니 주의하셔야 해요. 꽃노랑총채벌레를 방제하려면, 먼저 감염된 식물을 제거하고, 필요하면 비료와 물로 건강한 상태를 유지해야 합니다. 또한, 천적을 이용한 자연 방제 방법이나, 필요시 약제를 사용하시는 것도 좋은 방법입니다.',
      '담배가루이': '담배가루이는 작고 흰 가루 같아 보여요. 잎 뒷면에 붙어 즙을 빨아먹어 잎이 누렇게 시들고 끈적거릴 수 있으니 주의해야 해요. 방제하려면 먼저 부드러운 물줄기로 잎을 씻어 알과 벌레를 떨어뜨리고, 비눗물을 뿌려 남은 개체를 제거하세요. 필요하면 작은 기생벌 같은 천적을 방사하거나, 안전 기준에 지켜 농약을 살포해도 괜찮습니다.',
      '비단노린재': '비단노린재는 주로 잎이나 열매의 수액을 빨아먹는 해충으로, 식물에 큰 피해를 줄 수 있습니다. 이 해충이 발생하면 식물이 시들거나 열매가 썩을 수 있으니 주의하셔야 해요. 방제 방법으로는, 살충제를 사용하거나, 손으로 잡아 없애는 방법이 있습니다. 또, 정기적으로 온실을 점검해서 미리 예방하는 것이 중요합니다.',
      '알락수염노린재': '알락수염노린재는 주로 식물의 즙을 빨아먹는 해충이에요. 이 해충이 많으면 식물의 성장에 영향을 주고, 작물의 수확량이 줄어들 수 있으니 주의가 필요해요. 방제 방법으로는 먼저 유해한 부위를 잘 제거하고, 필요시 농약을 사용하는 것도 좋은 방법이랍니다. 주변 환경을 깨끗하게 유지하면 알락수염노린재의 서식을 줄일 수 있으니 참고하세요.'
    };

    return messages[insectName] || '해충이 탐지되었습니다. 정확한 분석을 위해 잠시만 기다려주세요.';
  };



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
    onMarkAsRead?.(alertDetail?.anlsIdx);
  };



  return (

    <div className="detection-feedback">

      {/* 백구의 탐지 분석 결과 텍스트 */}
      <div className="baekgu-msg-wrap">
        <div className="thumb">
          <img src="/images/talk_109.png" alt="" />
        </div>
        <div className="baekgu-msg">{alertDetail?.gptResult?.gptContent || getDefaultMessage(alertDetail?.greenhouseInfo?.insectName)}</div>
      </div>  

      <div className="baekgu-msg-wrap feedback">
        <div className="baekgu-msg">
          <textarea name="feedback_content" id="feedback_content" className='scrl-custom' placeholder="백구에게 전달하고 싶은 말이나 분석 결과에 대해 기록하고 싶은 말을 자유롭게 남겨주세요. 자세한 의견은 백구의 성장에 도움이 돼요."></textarea>
        </div>
      </div>     

      {step === 1 && (
       
        <div className="feedback-step">
          <div className="feedback-question">

            <div className="feedback-header">
              <h3 className="tit-2">해충 탐지 결과가 정확한가요?</h3>
            </div>

            <p className="question-text">
              <strong>{alertDetail?.greenhouseInfo?.insectName}</strong>을(를) <strong>구역</strong>에서 탐지했다고 하는데, 맞나요?
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

          </div>
        </div>
      )}

      {step === 3 && (
        <div className="feedback-complete">
          <h3 className="complete-title"><span className="complete-icon">✅</span> 피드백이 완료되었습니다!</h3>
          <p className="complete-message">
            소중한 의견 감사합니다. 백구가 더 똑똑해질 수 있도록 도와주셨어요!
          </p>
        </div>
      )}
    </div>
  );
}
