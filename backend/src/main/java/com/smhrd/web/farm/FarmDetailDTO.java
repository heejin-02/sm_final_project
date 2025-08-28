package com.smhrd.web.farm;

import lombok.Data;

@Data
public class FarmDetailDTO {
    private Long farmIdx;
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String farmCrops;
    private String farmArea;
    private String farmImg;
    
    // 유저 정보 포함 필요 시
    private String userPhone;
    private String userName;
}
