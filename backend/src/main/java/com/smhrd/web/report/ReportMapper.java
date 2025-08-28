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
    Integer getMonthlyTotalDetectionCount(@Param("farmIdx") Long farmIdx, @Param("month") String month);
    Integer getMonthlyInsectTypeCount(@Param("farmIdx") Long farmIdx, @Param("month") String month);
    String getMonthlyTopDetectionZone(@Param("farmIdx") Long farmIdx, @Param("month") String month);
    List<InsectMonthlyPredictionDTO> getMonthlyInsectPrediction(@Param("yearMonth") String yearMonth,
            @Param("prevYearMonth") String prevYearMonth);
    List<WeekCountDTO> getMonthlyWeeklyDetectionStats(@Param("farmIdx") Long farmIdx, @Param("month") String month);
    List<InsectDistributionDTO> getMonthlyInsectDistribution(@Param("farmIdx") Long farmIdx, @Param("month") String month);
    List<DetectionDetailDTO> getMonthlyDetectionDetails(@Param("farmIdx") Long farmIdx, @Param("month") String month);


    // 연별 통계
    Integer getYearlyTotalDetectionCount(@Param("farmIdx") Long farmIdx, @Param("year") String year);
    Integer getYearlyInsectTypeCount(@Param("farmIdx") Long farmIdx, @Param("year") String year);
    String getYearlyTopDetectionZone(@Param("farmIdx") Long farmIdx, @Param("year") String year);
    List<MonthCountDTO> getYearlyMonthlyStats(@Param("farmIdx") Long farmIdx, @Param("year") String year);
    List<InsectDistributionDTO> getYearlyInsectDistribution(@Param("farmIdx") Long farmIdx, @Param("year") String year);
    List<DetectionDetailDTO> getYearlyDetectionDetails(@Param("farmIdx") Long farmIdx, @Param("year") String year);

    // 4종 해충 연도별 탐지수
    List<InsectYearlyCountDTO> getYearlyInsectTrends(@Param("farmIdx") Long farmIdx, @Param("year") String year);

    // 4종 해충 연도별 탐지수
    List<InsectYearlyCountDTO> getYearlyInsectTrends(@Param("farmIdx") Long farmIdx, 
                                                      @Param("year") String year, 
                                                      @Param("prevYear") String prevYear);

    // 계절별 해충 예측 추가
    List<SeasonalPredictionDTO> getYearlySeasonalPrediction(
        @Param("farmIdx") Long farmIdx, 
        @Param("year") String year, 
        @Param("nextYear") String nextYear
    );
}