package com.smhrd.web.report;

import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ReportMapper {
    List<ReportDTO> findTodayReports(@Param("farmIdx") Long farmIdx); // 이건 그대로 사용 가능

    List<ReportDTO> findDailyReports(Map<String, Object> params);
    List<ReportDTO> findMonthlyReports(Map<String, Object> params);
    List<ReportDTO> findYearlyReports(Map<String, Object> params);
}
