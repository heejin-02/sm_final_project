package com.smhrd.web.farm.alert.dto;

import lombok.Data;

@Data
public class FeedbackDTO {
	private Long feedbackIdx;
    private Long anlsIdx;
	private String feedbackContent;
	private String createdAt;

}
