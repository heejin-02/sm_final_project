package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class InsectYearlyCountDTO {
    private String insectName;
    private int count2024;
    private int count2025;
    private int predicted2026;
}
