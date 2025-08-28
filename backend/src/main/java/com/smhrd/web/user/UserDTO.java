package com.smhrd.web.user;

import lombok.Data;

@Data
public class UserDTO {
	// Farm
	private Long farmIdx;
	private String farmName;
	private String farmAddr;
	private String farmPhone;
	private String farmCrops;
	private String farmArea;
	private String farmImg;
	
	// User
	private String userPhone;
	private String userName;

}

