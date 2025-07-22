package com.smhrd.web.admin;

import lombok.Data;

@Data
public class AdminDTO {
    private int farmIdx;
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String farmCrops;
    private String farmArea;
    private String createdAt;
    private String farmImg;
    private String userPhone;
}
