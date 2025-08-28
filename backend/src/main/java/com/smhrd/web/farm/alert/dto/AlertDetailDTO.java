package com.smhrd.web.farm.alert.dto;

import java.util.List;

import lombok.Data;

@Data
public class AlertDetailDTO {
    private GreenhouseInfoDTO greenhouseInfo;
    private List<ImageInfoDTO> imageList;
    private GptResultDTO gptResult;
    private FeedbackDTO feedback;
}
