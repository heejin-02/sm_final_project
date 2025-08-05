package com.smhrd.web.report.dto;

import lombok.Data;
import java.util.List;

@Data
public class MonthlyStatsDTO {
    private int totalCount;
    private int insectTypeCount;
    private String topZone;
    private List<DayCountDTO> dailyStats;
    private List<InsectDistributionDTO> insectDistribution;
    private List<ZoneCountDTO> zoneStats;
    private List<DetectionDetailDTO> details;


}
