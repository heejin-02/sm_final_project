package com.smhrd.web.report.dto;

import lombok.Data;
import java.util.List;

@Data
public class MonthlyStatsDTO {
    private int totalCount;                  // 총 탐지수
    private int insectTypeCount;             // 탐지된 해충 종류 수
    private String topZone;                  // 최다 탐지 구역 (예: "1번 온실(12건)")
    private List<InsectMonthlyPredictionDTO> predictions;   // 💡 내년 예측 데이터 리스트
    private List<WeekCountDTO> weeklyStats; // 주차별 탐지 현황 (1주차~4주차)
    private List<InsectDistributionDTO> insectDistribution; // 해충 종류별 분포 (원그래프용)
    private List<DetectionDetailDTO> details; // 상세 현황 (시간, 구역, 해충, 정확도)
}

