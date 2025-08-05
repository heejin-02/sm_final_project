// React 훅으로 상태 관리·페칭 로직 감싸기
import { useState, useEffect } from "react";
import { fetchRegionCounts } from "../api/regionApi";

export function useRegionCounts(farmId) {
  const [counts, setCounts] = useState(null);

  useEffect(() => {
    fetchRegionCounts(farmId)
      .then(setCounts)
      .catch((error) => {
        // console.error(error);
      });
  }, [farmId]);

  return counts;  // null(로딩) 또는 [{id, count},…]
}
