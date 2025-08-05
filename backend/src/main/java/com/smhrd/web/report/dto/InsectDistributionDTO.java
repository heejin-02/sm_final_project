package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class InsectDistributionDTO {
	// 해충 종류별 개수 (원그래프용)
    private String insect;
    private int count;
}
