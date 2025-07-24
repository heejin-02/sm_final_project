package com.smhrd.web.admin;

import lombok.Data;

@Data
public class AdminDTO {
    private int farmIdx;
    
    private String userName;
    private String userPhone;
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String joinAt;
    
}
