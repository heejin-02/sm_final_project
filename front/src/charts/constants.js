// src/charts/constants.js
export const INSECT_ORDER = [
  '꽃노랑총채벌레',
  '담배가루이',
  '비단노린재',
  '알락수염노린재',
];

// 사이트 톤에 맞게 살짝 밝힌 Soft 팔레트
export const COLORS = [
  '#5A8BD6', // blue
  '#B5D5F5', // light blue
  '#6FBC6B', // green
  '#9FD98D', // light green
  '#F39A3D', // orange
  '#FFC17A', // peach
  '#E86A6C', // coral red
  '#FFA3A1', // light coral
  '#BC86B8', // violet
  '#E0BEE0', // lavender
];

// 이름 공백 등 정규화
export const normalizeInsect = (s) =>
  s == null ? '' : String(s).trim();

// 이름 → 고정 색 매핑 (우선 지정 4종)
export const INSECT_COLOR = Object.fromEntries(
  INSECT_ORDER.map((name, i) => [name, COLORS[i % COLORS.length]])
);

// 현재 데이터에 있는 이름들을 “우선순위(INSECT_ORDER) + 나머지(가나다)”로 정렬
export const orderInsects = (names) => {
  const arr = Array.from(new Set((names || []).filter(Boolean)));
  const head = INSECT_ORDER.filter(n => arr.includes(n));
  const tail = arr.filter(n => !INSECT_ORDER.includes(n))
                  .sort((a, b) => a.localeCompare(b, 'ko'));
  return [...head, ...tail];
};
