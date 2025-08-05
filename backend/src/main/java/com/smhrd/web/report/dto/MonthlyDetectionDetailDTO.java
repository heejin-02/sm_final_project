package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class MonthlyDetectionDetailDTO {
    private String week;        // "1주차", "2주차" 등
    private String greenhouse;  // 온실 이름
    private int accuracy;       // 정확도
    
}
