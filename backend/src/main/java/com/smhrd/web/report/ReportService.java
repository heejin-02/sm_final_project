package com.smhrd.web.report;

import com.smhrd.web.report.dto.DailyStatsDTO;
import com.smhrd.web.report.dto.MonthlyStatsDTO;
import com.smhrd.web.report.dto.YearlyStatsDTO;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ReportService {

    @Autowired
    private ReportMapper reportMapper;

    public DailyStatsDTO getDailyStats(Long farmIdx, String date) {
        DailyStatsDTO dto = new DailyStatsDTO();

        dto.setTotalCount(reportMapper.getDailyTotalDetectionCount(farmIdx, date));
        dto.setInsectTypeCount(reportMapper.getDailyInsectTypeCount(farmIdx, date));
        dto.setTopZone(reportMapper.getDailyTopDetectionZone(farmIdx, date));
        dto.setHourlyStats(reportMapper.getDailyHourlyDetectionStats(farmIdx, date));
        dto.setInsectDistribution(reportMapper.getDailyInsectDistribution(farmIdx, date));
        dto.setZoneStats(reportMapper.getDailyZoneDetectionCounts(farmIdx, date));
        dto.setDetails(reportMapper.getDailyDetectionDetails(farmIdx, date));

        return dto;
    }

    public MonthlyStatsDTO getMonthlyStats(Long farmIdx, String month) {
        MonthlyStatsDTO dto = new MonthlyStatsDTO();

        dto.setTotalCount(reportMapper.getMonthlyTotalDetectionCount(farmIdx, month));
        dto.setInsectTypeCount(reportMapper.getMonthlyInsectTypeCount(farmIdx, month));
        dto.setTopZone(reportMapper.getMonthlyTopDetectionZone(farmIdx, month));
        dto.setWeeklyStats(reportMapper.getMonthlyWeeklyDetectionStats(farmIdx, month));
        dto.setInsectDistribution(reportMapper.getMonthlyInsectDistribution(farmIdx, month));
        dto.setDetails(reportMapper.getMonthlyDetectionDetails(farmIdx, month));

        return dto;
    }

    public YearlyStatsDTO getYearlyStats(Long farmIdx, String year) {
        YearlyStatsDTO dto = new YearlyStatsDTO();

        dto.setTotalCount(reportMapper.getYearlyTotalDetectionCount(farmIdx, year));
        dto.setInsectTypeCount(reportMapper.getYearlyInsectTypeCount(farmIdx, year));
        dto.setTopZone(reportMapper.getYearlyTopDetectionZone(farmIdx, year));
        dto.setMonthlyStats(reportMapper.getYearlyMonthlyStats(farmIdx, year));
        dto.setInsectDistribution(reportMapper.getYearlyInsectDistribution(farmIdx, year));
        dto.setDetails(reportMapper.getYearlyDetectionDetails(farmIdx, year));
        dto.setPredictedInsectTrends(reportMapper.getYearlyInsectTrends(farmIdx));

        return dto;
    }

}

