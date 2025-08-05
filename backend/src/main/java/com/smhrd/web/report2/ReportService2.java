package com.smhrd.web.report2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ReportService2 {

    @Autowired
    private ReportMapper2 reportMapper;

    public List<ReportDTO2> getTodayReports(Long farmIdx) {
        return reportMapper.findTodayReports(farmIdx);
    }

    public List<ReportDTO2> getDailyReports(Long farmIdx, String year, String month, String day) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        params.put("month", month);
        params.put("day", day);
        return reportMapper.findDailyReports(params);
    }

    public List<ReportDTO2> getMonthlyReports(Long farmIdx, String year, String month) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        params.put("month", month);
        return reportMapper.findMonthlyReports(params);
    }

    public List<ReportDTO2> getYearlyReports(Long farmIdx, String year) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        return reportMapper.findYearlyReports(params);
    }
}
