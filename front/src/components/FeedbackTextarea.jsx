import { useState, useEffect } from 'react';
import { API } from '../api/http'; // axios 인스턴스
import AlertModal from './AlertModal';

export default function FeedbackTextarea({ anlsIdx, alertDetail }) {
  const [content, setContent] = useState('');
  const [savedFeedback, setSavedFeedback] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');

  // 초기 피드백 불러오기
  useEffect(() => {
    if (!anlsIdx) return;

    API.get(`/user/alert/feedback/${anlsIdx}`)
      .then((res) => {
        if (res.data?.feedbackContent) {
          setSavedFeedback(res.data);
          setContent(res.data.feedbackContent);
        }
      })
      .catch((err) => {
        console.warn('피드백 조회 실패:', err.message);
      });
  }, [anlsIdx]);

  // 등록 또는 수정 요청
  const handleSubmit = async () => {
    if (!content.trim()) return;
    setIsSubmitting(true);

    try {
      if (savedFeedback) {
        // 수정
        await API.put(`/user/alert/feedback/${anlsIdx}`, {
          feedbackIdx: savedFeedback.feedbackIdx,
          anlsIdx,
          feedbackContent: content,
          createdAt: new Date().toISOString(),
        });
        setAlertMessage('기록을 수정했습니다.');
      } else {
        // 등록
        const res = await API.post(`/user/alert/feedback`, {
          anlsIdx,
          feedbackContent: content,
          createdAt: new Date().toISOString(),
        });
        setAlertMessage('기록을 저장했습니다.');

        // 등록 후 상태 갱신
        setSavedFeedback({
          feedbackContent: content,
          feedbackIdx: res.data?.feedbackIdx || Date.now(),
        });
      }
      setIsEditing(false);
      setShowAlert(true);
    } catch (error) {
      console.error('피드백 저장 실패:', error);
      setAlertMessage('기록에 실패했습니다. 다시 시도해주세요.');
      setShowAlert(true);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <div className="baekgu-msg-wrap feedback">
        <div className="baekgu-msg">
          <textarea
            name="feedback_content"
            className="scrl-custom resize-none"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="백구에게 전달하고 싶은 말이나 분석 결과에 대해 기록하고 싶은 말을 자유롭게 남겨주세요."
            readOnly={!isEditing && !!savedFeedback}
          />

          <div className="btn-wrap">
            {!savedFeedback ? (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || !content.trim()}
                className="btn"
              >
                {isSubmitting ? '저장 중...' : '등록하기'}
              </button>
            ) : !isEditing ? (
              <button onClick={() => setIsEditing(true)} className="btn">
                수정하기
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || !content.trim()}
                className="btn"
              >
                {isSubmitting ? '저장 중...' : '수정 완료'}
              </button>
            )}
          </div>
        </div>
      </div>

      {showAlert && (
        <AlertModal message={alertMessage} onClose={() => setShowAlert(false)} />
      )}
    </>
  );
}
