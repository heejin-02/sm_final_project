package com.smhrd.web.userDetailDTO;

import java.math.BigDecimal;

import lombok.Data;

@Data
public class CCTVDTO {
	// CCTV
	private Long cctvIdx;
	private String cctvNmString;
	private String cctvLoc;
	private BigDecimal cctvX;
	private BigDecimal cctvY;
}
