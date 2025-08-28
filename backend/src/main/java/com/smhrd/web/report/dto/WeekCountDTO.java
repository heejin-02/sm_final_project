package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class WeekCountDTO {
    private String week;  // ex) "1주차", "2주차"
    private int count;    // 해당 주차 탐지 수
}
