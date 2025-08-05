package com.smhrd.web.report;

import com.smhrd.web.report.dto.DailyStatsDTO;
import com.smhrd.web.report.dto.MonthlyStatsDTO;

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


}

