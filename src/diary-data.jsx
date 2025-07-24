// src/diary-data.js

/**
 * @typedef {Object} Diary
 * @property {string} id
 * @property {string} title
 * @property {'기쁨'|'슬픔'|'분노'|'피곤'} [mood]
 * @property {string} date
 * @property {string} content
 */

/** @type {Diary[]} */
export const diaryData = (() => {
  try {
    const raw = localStorage.getItem('diaryData');
    return raw ? JSON.parse(raw) : [];
  } catch {
    console.warn('diaryData parsing failed, initializing empty array');
    return [];
  }
})();

/** 저장 함수: diaryData를 localStorage에 반영 */
export function saveDiaryData() {
  try {
    localStorage.setItem('diaryData', JSON.stringify(diaryData));
  } catch (e) {
    console.error('Failed to save diaryData', e);
  }
}
