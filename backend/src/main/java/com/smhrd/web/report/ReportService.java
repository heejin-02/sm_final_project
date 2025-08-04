package com.smhrd.web.report;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ReportService {

    @Autowired
    private ReportMapper reportMapper;

    public List<ReportDTO> getTodayReports(Long farmIdx) {
        return reportMapper.findTodayReports(farmIdx);
    }

    public List<ReportDTO> getDailyReports(Long farmIdx, String year, String month, String day) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        params.put("month", month);
        params.put("day", day);
        return reportMapper.findDailyReports(params);
    }

    public List<ReportDTO> getMonthlyReports(Long farmIdx, String year, String month) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        params.put("month", month);
        return reportMapper.findMonthlyReports(params);
    }

    public List<ReportDTO> getYearlyReports(Long farmIdx, String year) {
        Map<String, Object> params = new HashMap<>();
        params.put("farmIdx", farmIdx);
        params.put("year", year);
        return reportMapper.findYearlyReports(params);
    }
}
