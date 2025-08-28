package com.smhrd.web.report.dto;

import lombok.Data;
import java.util.List;

@Data
public class YearlyStatsDTO {
    private int totalCount;
    private int insectTypeCount;
    private String topZone;
    private List<SeasonalPredictionDTO> predictedInsectTrends;  // List<SeasonalPredictionDTO>로 수정

    private List<MonthCountDTO> monthlyStats;
    private List<InsectDistributionDTO> insectDistribution;
    private List<DetectionDetailDTO> details;
}
