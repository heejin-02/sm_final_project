package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class WeekCountDTO {
    private String week; // 예: "1주차", "2주차"
    private int count;
}
