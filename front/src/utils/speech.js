// utils/speech.js
export function speak(text, { rate = 0.7, pitch = 1, voiceName } = {}) {
  if (!window.speechSynthesis) {
    alert('이 브라우저는 음성 안내를 지원하지 않습니다.');
    return;
  }

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = 'ko-KR';
  utter.rate = rate; // 말하기 속도 (0.5 ~ 2)
  utter.pitch = pitch; // 목소리 톤 (0 ~ 2)

  // 브라우저의 음성 목록 가져오기
  const voices = window.speechSynthesis.getVoices();

  // voiceName으로 선택하거나, ko-KR 기본 음성 선택
  if (voiceName) {
    const selected = voices.find((v) => v.name.includes(voiceName));
    if (selected) utter.voice = selected;
  } else {
    const korean = voices.find((v) => v.lang === 'ko-KR');
    if (korean) utter.voice = korean;
  }

  window.speechSynthesis.speak(utter);
}

// 사용 가능한 음성 리스트 확인용
export function listVoices() {
  return window.speechSynthesis.getVoices();
}
