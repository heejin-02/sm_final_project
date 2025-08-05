package com.smhrd.web.report2;

import lombok.Data;
import java.sql.Timestamp;

@Data
public class ReportDTO2 {
    private Long reportIdx;
    private Long farmIdx;
    private String periodType;
    private String periodMark;
    private String report;
    private Timestamp createdAt; // CREATED_AT (Timestamp로 받는 게 좋음)
    private Long gptIdx;
}
