import { useState, useEffect } from 'react';
import axios from 'axios';
import AlertModal from './AlertModal';

export default function FeedbackTextarea({ anlsIdx, alertDetail }) {
  const [content, setContent] = useState('');
  const [savedFeedback, setSavedFeedback] = useState(null);
  const [isEditing, setIsEditing] = useState(false);  // 수정 상태로 시작
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showAlert, setShowAlert] = useState(false);  // 알림 모달 표시 여부
  const [alertMessage, setAlertMessage] = useState('');  // 알림 메시지

  // 초기 피드백 불러오기 (anlsIdx가 존재할 때만 실행)
  useEffect(() => {
    if (!anlsIdx) return;

    axios
      .get(`http://localhost:8095/user/alert/feedback/${anlsIdx}`)  // 경로 변수로 요청
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
        // PUT 요청으로 수정
        await axios.put(`http://localhost:8095/user/alert/feedback/${anlsIdx}`, {
          feedbackIdx: savedFeedback.feedbackIdx,
          anlsIdx,
          feedbackContent: content,
          createdAt: new Date().toISOString(),
        });
        setAlertMessage('기록을 수정했습니다.');
      } else {
        // POST 요청으로 등록
        await axios.post(`http://localhost:8095/user/alert/feedback`, {
          anlsIdx,
          feedbackContent: content,
          createdAt: new Date().toISOString(),
        });
        setAlertMessage('기록을 저장했습니다.');
        
        // 등록 후 상태 업데이트 (피드백 데이터로 갱신)
        setSavedFeedback({
          feedbackContent: content,
          feedbackIdx: Date.now(),  // 임시 피드백 ID (응답에서 실제 ID로 수정 필요)
        });
      }
      setIsEditing(false);  // 수정 후 읽기 전용 모드로 전환
      setShowAlert(true); 
    } catch (error) {
      console.error('피드백 저장 실패:', error);
      setAlertMessage('기록에 실패했습니다. 다시 시도해주세요.');
      setShowAlert(true); 
    } finally {
      setIsSubmitting(false);
      setShowAlert(true); 
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
            readOnly={!isEditing && !!savedFeedback}  // 데이터가 있으면 읽기 전용
          />
          
          <div className="btn-wrap">
            {!savedFeedback ? (
              // 데이터가 없으면 등록 버튼을 보여준다
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || !content.trim()}
                className="btn"
              >
                {isSubmitting ? '저장 중...' : '등록하기'}
              </button>
            ) : !isEditing ? (
              // 데이터가 있으면 수정 버튼을 보여준다
              <button
                onClick={() => setIsEditing(true)}  // 수정하기 버튼 클릭 시 수정 모드로 전환
                className="btn"
              >
                수정하기
              </button>
            ) : (
              // 수정 모드에서 수정 완료 버튼을 보여준다
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

      {/* 알림 모달 */}
      {showAlert && (
        <AlertModal message={alertMessage} onClose={() => setShowAlert(false)} />
      )}

    </>

  );
}
