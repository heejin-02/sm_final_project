package com.smhrd.web.admin;

import java.util.Date;
import lombok.Data;

@Data
public class AdminDTO {
	private Long farmIdx;

    private String userName;
    private String userPhone;
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private Date createdAt;

    private String userPw;
    private String joinedAt; 
    private String farmCrops;
    private String farmArea;
}
