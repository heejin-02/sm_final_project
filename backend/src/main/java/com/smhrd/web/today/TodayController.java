package com.smhrd.web.today;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/user/today")
public class TodayController {

    @Autowired
    private TodayService todayService;
    
    @Operation(summary = "오늘 찾은 해충 수")
    @GetMapping("/today")
    public TodayResultDTO getTodayResult() {
        return todayService.getTodayResult();
    }
}
