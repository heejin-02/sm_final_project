// utils/groupDataByWeek.js
export function groupDataByWeek(details) {
  const grouped = {};

  details.forEach((item) => {
    const date = new Date(item.time);
    const week = Math.ceil(date.getDate() / 7); // 1~5주차
    const key = `${date.getFullYear()}-${date.getMonth() + 1}-${week}`;

    if (!grouped[key]) {
      grouped[key] = {
        title: `${date.getFullYear()}년 ${date.getMonth() + 1}월 ${week}주차`,
        count: 0,
        items: []
      };
    }

    grouped[key].items.push({
      time: item.time,
      greenhouse: item.greenhouse,
      insect: item.insect,
      accuracy: item.accuracy,
      anlsIdx: item.anlsIdx
    });
    grouped[key].count += 1;
  });

  return grouped;
}
