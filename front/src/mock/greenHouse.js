// mock/greenHouse.js
// 구역(온실) 더미 데이터

// 기본 구역 생성 함수 (1~9번 구역)
export const generateDefaultRegions = (farmIdx) => {
  const regions = [];
  for (let i = 1; i <= 9; i++) {
    regions.push({
      ghIdx: i,
      farmIdx: farmIdx,
      ghName: `${i}번 구역`,
      ghArea: '1000',
      ghCrops: '토마토',
      createdAt: new Date().toISOString()
    });
  }
  return regions;
};

// 구역 더미 데이터 (여러 농장용)
export const GH_DUMMY = [
  // 농장 1의 구역들
  { ghIdx: 1, farmIdx: 1, ghName: '1번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-21 01:56:35' },
  { ghIdx: 2, farmIdx: 1, ghName: '2번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-20 01:56:35' },
  { ghIdx: 3, farmIdx: 1, ghName: '3번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-19 01:56:35' },
  { ghIdx: 4, farmIdx: 1, ghName: '4번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-18 01:56:35' },
  { ghIdx: 5, farmIdx: 1, ghName: '5번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-17 01:56:35' },
  { ghIdx: 6, farmIdx: 1, ghName: '6번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-16 01:56:35' },
  { ghIdx: 7, farmIdx: 1, ghName: '7번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-15 01:56:35' },
  { ghIdx: 8, farmIdx: 1, ghName: '8번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-14 01:56:35' },
  { ghIdx: 9, farmIdx: 1, ghName: '9번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-13 01:56:35' },

  // 농장 2의 구역들 (DB 데이터 예시)
  { ghIdx: 1, farmIdx: 2, ghName: '1번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-21 01:56:35' },
  { ghIdx: 2, farmIdx: 2, ghName: '2번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-20 01:56:35' },
  { ghIdx: 3, farmIdx: 2, ghName: '3번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-19 01:56:35' },
  { ghIdx: 4, farmIdx: 2, ghName: '4번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-18 01:56:35' },
  { ghIdx: 5, farmIdx: 2, ghName: '5번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-17 01:56:35' },
  { ghIdx: 6, farmIdx: 2, ghName: '6번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-16 01:56:35' },
  { ghIdx: 7, farmIdx: 2, ghName: '7번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-15 01:56:35' },
  { ghIdx: 8, farmIdx: 2, ghName: '8번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-14 01:56:35' },
  { ghIdx: 9, farmIdx: 2, ghName: '9번 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-13 01:56:35' },

  // 농장 3의 구역들 (일부만 DB에 있는 경우 예시)
  { ghIdx: 1, farmIdx: 3, ghName: '메인 구역', ghArea: '1000', ghCrops: '오이', createdAt: '2025-07-21 01:56:35' },
  { ghIdx: 3, farmIdx: 3, ghName: '실험 구역', ghArea: '1000', ghCrops: '상추', createdAt: '2025-07-19 01:56:35' },
  { ghIdx: 5, farmIdx: 3, ghName: '육묘 구역', ghArea: '1000', ghCrops: '토마토', createdAt: '2025-07-17 01:56:35' },
];

// 특정 농장의 구역 정보 조회 (DB + 기본값 병합)
export const getRegionsByFarmIdx = (farmIdx) => {
  // DB에서 해당 농장의 구역 정보 조회
  const dbRegions = GH_DUMMY.filter(region => region.farmIdx === farmIdx);
  
  // 1~9번 구역 생성 (DB에 없는 구역은 기본값 사용)
  const allRegions = [];
  for (let i = 1; i <= 9; i++) {
    const dbRegion = dbRegions.find(region => region.ghIdx === i);
    
    if (dbRegion) {
      // DB에 있는 구역은 DB 데이터 사용
      allRegions.push(dbRegion);
    } else {
      // DB에 없는 구역은 기본값 사용
      allRegions.push({
        ghIdx: i,
        farmIdx: farmIdx,
        ghName: `${i}번 구역`,
        ghArea: '1000',
        ghCrops: '토마토',
        createdAt: new Date().toISOString()
      });
    }
  }
  
  return allRegions;
};

// FarmMap용 구역 데이터 (탐지 개수 포함)
export const getRegionsWithDetectionCount = (farmIdx) => {
  const regions = getRegionsByFarmIdx(farmIdx);

  // 각 구역에 더미 탐지 개수 추가
  return regions.map(region => ({
    ...region,
    count: Math.floor(Math.random() * 20), // 0~19개 랜덤 탐지 개수
    gh_name: region.ghName // FarmMap에서 사용할 필드명
  }));
};

// 구역 정보 업데이트 (AdminFarmInfo에서 사용)
export const updateRegionsForFarm = (farmIdx, newRegions) => {
  // 기존 해당 농장 구역 제거
  const otherFarmRegions = GH_DUMMY.filter(region => region.farmIdx !== farmIdx);

  // 새 구역 데이터 추가
  GH_DUMMY.length = 0; // 배열 초기화
  GH_DUMMY.push(...otherFarmRegions, ...newRegions);

  console.log(`농장 ${farmIdx}의 구역 정보가 업데이트되었습니다:`, newRegions);
  return true;
};
