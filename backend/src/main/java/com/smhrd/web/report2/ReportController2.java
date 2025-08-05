package com.smhrd.web.report2;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/report")
@Tag(name = "Report Controller2", description = "통계 화면 API")
public class ReportController2 {

    @Autowired
    private ReportService2 reportService;

    @GetMapping("/today/{farmIdx}")
    @Operation(summary = "오늘 알림 통계입니다.")
    public List<ReportDTO2> getTodayReport(@PathVariable Long farmIdx) {
        return reportService.getTodayReports(farmIdx);
    }

    @GetMapping("/daily")
    @Operation(summary = "일별 통계입니다.")
    public List<ReportDTO2> getDailyReports(@RequestParam Long farmIdx,
                                           @RequestParam String year,
                                           @RequestParam String month,
                                           @RequestParam String day) {
        return reportService.getDailyReports(farmIdx, year, month, day);
    }


    @GetMapping("/monthly")
    @Operation(summary = "월별 통계입니다.")
    public List<ReportDTO2> getMonthlyReports(@RequestParam Long farmIdx,
                                             @RequestParam String year,
                                             @RequestParam String month) {
        return reportService.getMonthlyReports(farmIdx, year, month);
    }

    @GetMapping("/yearly")
    @Operation(summary = "연별 통계입니다.")
    public List<ReportDTO2> getYearlyReports(@RequestParam Long farmIdx,
                                            @RequestParam String year) {
        return reportService.getYearlyReports(farmIdx, year);
    }
}
