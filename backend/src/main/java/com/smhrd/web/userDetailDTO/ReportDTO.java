package com.smhrd.web.userDetailDTO;

import lombok.Data;

@Data
public class ReportDTO {
    private int reportIdx;
    private int farmIdx;
    private String reportTitle;
    private String reportContent;
    private String createdDate;

}
