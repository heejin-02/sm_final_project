package com.smhrd.web.admin;

import org.springframework.web.multipart.MultipartFile;
import lombok.Data;

@Data
public class FarmInsertDTO {
    private String userPhone;   // 사용자 전화번호 (FK)
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String farmCrops;
    private String farmArea;

    // 이미지 파일
    private MultipartFile farmImg;

    // DB에 저장할 이미지 파일명
    private String farmImgName;
}
