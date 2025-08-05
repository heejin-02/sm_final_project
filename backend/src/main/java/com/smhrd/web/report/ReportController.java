package com.smhrd.web.report;

import com.smhrd.web.report.dto.DailyStatsDTO;
import com.smhrd.web.report.dto.MonthlyStatsDTO;

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

    

}
