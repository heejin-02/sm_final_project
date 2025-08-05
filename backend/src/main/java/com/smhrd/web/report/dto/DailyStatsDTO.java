package com.smhrd.web.report.dto;

import lombok.Data;
import java.util.List;

@Data
public class DailyStatsDTO {
	// 일별 (전체 응답 DTO)
    private int totalCount;
    private int insectTypeCount;
    private String topZone;
    private List<TimeCountDTO> hourlyStats;
    private List<InsectDistributionDTO> insectDistribution;
    private List<ZoneCountDTO> zoneStats;
    private List<DetectionDetailDTO> details;
}
