package com.smhrd.web.farm.alert.dto;


import lombok.Data;

@Data
public class GreenhouseInfoDTO {
    private Long ghIdx;
    private Long farmIdx;
    private String ghName;
    private String ghArea;
    private String ghCrops;
    private String createdAt;
}
