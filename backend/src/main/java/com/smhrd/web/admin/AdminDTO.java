package com.smhrd.web.admin;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "Admin 농장 데이터 전송 객체")
@Data
public class AdminDTO {

    private int farmIdx;

    private String userName;
    private String userPhone;
    private String userPw;

    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String joinedAt;  // 가능하면 LocalDateTime으로 바꾸고 Jackson Date 포맷 설정 권장

    private String farmCrops;
    private String farmArea;

    private String farmImgName;  // DB에 저장된 이미지 파일명

    // MultipartFile은 REST API용 DTO에는 포함하지 않는게 좋음
    // 대신, 이미지 업로드는 별도 API에서 MultipartFile만 받는 것이 권장됨
}
