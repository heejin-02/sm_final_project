package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class TimeCountDTO {
    private String hour;  // 시간 (예: "10", "14")
    private int count;
    
}
