package com.smhrd.web.farm.alert.dto;

import lombok.Data;

@Data
public class GptResultDTO {
    private Long gptIdx;
    private String userQes;
    private String userFile1;
    private String userFile2;
    private String userFile3;
    private String gptContent;
    private String createdAt;
    private Long anlsIdx;
}
