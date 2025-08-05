package com.smhrd.web.report2;

import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ReportMapper2 {
    List<ReportDTO2> findTodayReports(@Param("farmIdx") Long farmIdx); // 이건 그대로 사용 가능

    List<ReportDTO2> findDailyReports(Map<String, Object> params);
    List<ReportDTO2> findMonthlyReports(Map<String, Object> params);
    List<ReportDTO2> findYearlyReports(Map<String, Object> params);
}
