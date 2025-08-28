package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class InsectMonthlyPredictionDTO {
    private String month;       // 예: "2025-08"
    private String insectName;  // 해충 이름
    private int count2024;      // 2024년 발생 수
    private int count2025;      // 2025년 발생 수
    private int predicted2026;  // 예측 값
}
