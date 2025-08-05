package com.smhrd.web.report;

import com.smhrd.web.report.dto.*;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface ReportMapper {

    // 일별 통계
    Integer getDailyTotalDetectionCount(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    Integer getDailyInsectTypeCount(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    String getDailyTopDetectionZone(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    List<TimeCountDTO> getDailyHourlyDetectionStats(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    List<InsectDistributionDTO> getDailyInsectDistribution(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    List<ZoneCountDTO> getDailyZoneDetectionCounts(@Param("farmIdx") Long farmIdx, @Param("date") String date);
    List<DetectionDetailDTO> getDailyDetectionDetails(@Param("farmIdx") Long farmIdx, @Param("date") String date);

    // 월별 통계
    
}