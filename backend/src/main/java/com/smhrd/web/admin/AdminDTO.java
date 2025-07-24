package com.smhrd.web.admin;

import org.springframework.web.multipart.MultipartFile;

import lombok.Data;

@Data
public class AdminDTO {
    private int farmIdx;
    
    private String userName;
    private String userPhone;
    private String userPw;

    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String joinedAt; 
    
    private String farmCrops;
    private String farmArea;

    // 이미지 실제 파일은 MultipartFile 타입으로 받기
    private MultipartFile farmImg;

    // DB에 저장할 파일명(경로) 따로 필요하면 필드 추가 가능
    private String farmImgName;
    
}
