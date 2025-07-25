package com.smhrd.web.userDetailDTO;

import java.sql.Date;
import lombok.Data;

@Data
public class GreenhouseDTO {
	// Greenhouse
	private Long ghIdx;
	private String ghName;
	private String ghArea;
	private String ghCrops;
	private Date createdAt;
}
