// GptSummary.jsx
export default function GptSummary({ gptSummary, gptLoading, gptError }) {
  return (
    <div className='baekgu-msg-wrap mt-8 flex'>
      <div className='thumb'>
        <img src='/images/talk_109.png' alt='백구' />
      </div>
      <div className='baekgu-msg w-full'>
        {gptLoading &&
          '통계 내용을 토대로 분석 중입니다. 잠시만 기다려 주세요.'}
        {gptError && '분석 요청이 실패했습니다.'}
        {!gptLoading && !gptError && (gptSummary || '분석을 준비 중입니다.')}
      </div>
    </div>
  );
}
