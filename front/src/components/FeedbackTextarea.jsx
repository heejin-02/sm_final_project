import { useState, useEffect } from 'react';
import axios from 'axios';

export default function FeedbackTextarea({ anlsIdx, alertDetail }) {
  const [content, setContent] = useState('');
  const [savedFeedback, setSavedFeedback] = useState(null);
  const [isEditing, setIsEditing] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 초기 피드백 불러오기 (anlsIdx가 존재할 때만 실행)
  useEffect(() => {
    if (!anlsIdx) return;

    axios
      .get(`http://localhost:8095/user/alert/feedback?anlsIdx=${anlsIdx}`)
      .then((res) => {
        if (res.data?.feedbackContent) {
          setSavedFeedback(res.data);
          setContent(res.data.feedbackContent);
          setIsEditing(false); // 읽기 전용 모드
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
      await axios.post(`http://localhost:8095/user/alert/feedback`, {
        feedbackIdx: savedFeedback?.feedbackIdx || null,
        anlsIdx,
        feedbackContent: content,
        createdAt: new Date().toISOString(),
      });

      alert(savedFeedback ? '피드백이 수정되었습니다.' : '피드백이 등록되었습니다.');
      setIsEditing(false);
    } catch (error) {
      console.error('피드백 저장 실패:', error);
      alert('피드백 저장에 실패했습니다.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="baekgu-msg-wrap feedback">
      <div className="baekgu-msg">
        <textarea
          name="feedback_content"
          className="scrl-custom resize-none"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="백구에게 전달하고 싶은 말이나 분석 결과에 대해 기록하고 싶은 말을 자유롭게 남겨주세요."
          readOnly={!isEditing}
        />

        <div className="btn-wrap">
          {isEditing ? (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !content.trim()}
              className="btn"
            >
              {isSubmitting ? '저장 중...' : savedFeedback ? '수정 완료' : '저장하기'}
            </button>
          ) : (
            <button
              onClick={() => setIsEditing(true)}
              className="btn"
            >
              수정하기
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
