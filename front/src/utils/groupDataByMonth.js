// utils/groupDataByMonth.js
export function groupDataByMonth(details) {
  const grouped = {};

  details.forEach((item) => {
    const date = new Date(item.time);
    const month = date.getMonth() + 1;
    const key = `${date.getFullYear()}-${month.toString().padStart(2, '0')}`;

    if (!grouped[key]) {
      grouped[key] = {
        title: `${date.getFullYear()}년 ${month}월`,
        count: 0,
        items: []
      };
    }

    grouped[key].items.push({
      datetime: item.time,
      region: item.greenhouse,
      bugType: item.insect,
      accuracy: item.accuracy,
      anlsIdx: item.anlsIdx ?? `${item.time}-${item.greenhouse}`
    });
    grouped[key].count += 1;
  });

  return grouped;
}
