package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class DayCountDTO {
    private String day;  // "29", "30" 일(day) 정보
    private int count;
}
