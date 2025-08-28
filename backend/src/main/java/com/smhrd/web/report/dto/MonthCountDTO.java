package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class MonthCountDTO {
    private String month; // 예: "01", "02"
    private int count;
}
