package com.smhrd.web.report.dto;

import lombok.Data;

@Data
public class DetectionDetailDTO {
   // 상세 리스트 (시간, 해충명, 정확도 등)
   private int anlsIdx;
    private String time;
    private String greenhouse;
    private String insect;
    private int accuracy;
}
