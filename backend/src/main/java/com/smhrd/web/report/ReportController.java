package com.smhrd.web.report;

import com.smhrd.web.report.dto.DailyStatsDTO;
import com.smhrd.web.report.dto.MonthlyStatsDTO;
import com.smhrd.web.report.dto.YearlyStatsDTO;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/report")
@Tag(name = "Report Controller", description = "통계 알림 관련 API")
public class ReportController {

    @Autowired
    private ReportService reportService;

    @GetMapping("/daily-stats")
    @Operation(summary = "일간 탐지 통계 조회")
    public DailyStatsDTO getDailyStats(@RequestParam Long farmIdx,
                                       @RequestParam String date) {
        return reportService.getDailyStats(farmIdx, date);
    }

    
    @GetMapping("/monthly-stats")
    @Operation(summary = "월간 탐지 통계 조회")
    public MonthlyStatsDTO getMonthlyStats(@RequestParam Long farmIdx,
                                           @RequestParam String month) {
        return reportService.getMonthlyStats(farmIdx, month);
    }
    
    @GetMapping("/yearly-stats")
    @Operation(summary = "연간 탐지 통계 조회")
    public YearlyStatsDTO getYearlyStats(@RequestParam Long farmIdx,
                                         @RequestParam String year) {
        return reportService.getYearlyStats(farmIdx, year);
    }

}
