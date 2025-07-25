package com.smhrd.web.userDetailDTO;

import lombok.Data;

@Data
public class FeedbackDTO {
    private int feedbackIdx;
    private int farmIdx;
    private String managerName;
    private String content;
    private String regDate;
}
