package com.smhrd.web.today;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/user/today")
@Tag(name = "Today Controller", description = "오늘 찾은 해충관련 API")
public class TodayController {

    @Autowired
    private TodayService todayService;
    
    @GetMapping("/today")
    @Operation(summary = "오늘 찾은 해충 수, 종류, 구역 입니다. (농장 기준)")
    public TodayResultDTO getTodayResultByFarm(@RequestParam("farmIdx") Long farmIdx) {
        return todayService.getTodayResultByFarm(farmIdx);
    }
    
    @GetMapping("/today/greenhouses")
    @Operation(summary = "오늘 농장의 구역별 벌레 수를 확인합니다.")
    public List<TodayGreenhouseDTO> getTodayResultByGreenhouse(@RequestParam("farmIdx") Long farmIdx) {
        return todayService.getTodayResultByGreenhouse(farmIdx);
    }
    
    @GetMapping("/today/gptReports")
    @Operation(summary = "오늘 농장의 GPT 리포트 리스트를 조회합니다.")
    public List<String> getTodayGptReportsByFarm(@RequestParam("farmIdx") Long farmIdx) {
        return todayService.getTodayGptReportsByFarm(farmIdx);
    }


}
