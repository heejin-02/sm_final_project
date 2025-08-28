package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class ZoneCountDTO {
	//	구역별 탐지 수
    private String zone;
    private int count;
}
