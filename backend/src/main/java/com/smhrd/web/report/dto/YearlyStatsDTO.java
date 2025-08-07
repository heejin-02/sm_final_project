package com.smhrd.web.report.dto;

import java.util.List;

import lombok.Data;

@Data
public class YearlyStatsDTO {
    private int totalCount;
    private int insectTypeCount;
    private String topZone;
    private List<InsectYearlyCountDTO> predictedInsectTrends;
    private List<MonthCountDTO> monthlyStats;
    private List<InsectDistributionDTO> insectDistribution;
    private List<DetectionDetailDTO> details;
}
