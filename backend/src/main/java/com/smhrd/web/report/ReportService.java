package com.smhrd.web.report;

import com.smhrd.web.report.dto.DailyStatsDTO;
import com.smhrd.web.report.dto.InsectMonthlyPredictionDTO;
import com.smhrd.web.report.dto.MonthlyStatsDTO;
import com.smhrd.web.report.dto.SeasonalPredictionDTO;
import com.smhrd.web.report.dto.WeekCountDTO;
import com.smhrd.web.report.dto.YearlyStatsDTO;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

        List<WeekCountDTO> rawWeeklyStats = reportMapper.getMonthlyWeeklyDetectionStats(farmIdx, month);
        Map<String, Integer> weekMap = new LinkedHashMap<>();
        weekMap.put("1주차", 0);
        weekMap.put("2주차", 0);
        weekMap.put("3주차", 0);
        weekMap.put("4주차", 0);

        for (WeekCountDTO dtoItem : rawWeeklyStats) {
            weekMap.put(dtoItem.getWeek(), dtoItem.getCount());
        }

        List<WeekCountDTO> completedStats = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : weekMap.entrySet()) {
            WeekCountDTO weekDTO = new WeekCountDTO();
            weekDTO.setWeek(entry.getKey());
            weekDTO.setCount(entry.getValue());
            completedStats.add(weekDTO);
        }

        dto.setWeeklyStats(completedStats);

        dto.setInsectDistribution(reportMapper.getMonthlyInsectDistribution(farmIdx, month));
        dto.setDetails(reportMapper.getMonthlyDetectionDetails(farmIdx, month));
     // 이전 년도 same month 구하기 (ex: 2025-07 -> 2024-07)
        String[] parts = month.split("-");
        int year = Integer.parseInt(parts[0]);
        int mon = Integer.parseInt(parts[1]);
        int prevYear = year - 1;
        String prevYearMonth = String.format("%04d-%02d", prevYear, mon);

        // 매퍼에서 이전 년도와 현재 년도 데이터를 기준으로 예측값 계산해서 가져오기
        List<InsectMonthlyPredictionDTO> predictions = reportMapper.getMonthlyInsectPrediction(month, prevYearMonth);
        dto.setPredictions(predictions);

        return dto;
    }


    public YearlyStatsDTO getYearlyStats(Long farmIdx, String year) {
        YearlyStatsDTO dto = new YearlyStatsDTO();

        // 기존 통계 데이터 설정
        dto.setTotalCount(reportMapper.getYearlyTotalDetectionCount(farmIdx, year));
        dto.setInsectTypeCount(reportMapper.getYearlyInsectTypeCount(farmIdx, year));
        dto.setTopZone(reportMapper.getYearlyTopDetectionZone(farmIdx, year));
        dto.setMonthlyStats(reportMapper.getYearlyMonthlyStats(farmIdx, year));
        dto.setInsectDistribution(reportMapper.getYearlyInsectDistribution(farmIdx, year));
        dto.setDetails(reportMapper.getYearlyDetectionDetails(farmIdx, year));
        
        // 계절별 해충 예측 설정
        List<SeasonalPredictionDTO> seasonalPredictions = reportMapper.getYearlySeasonalPrediction(farmIdx, year, String.valueOf(Integer.parseInt(year) + 1));
        dto.setPredictedInsectTrends(seasonalPredictions);

        return dto;
    }




}

