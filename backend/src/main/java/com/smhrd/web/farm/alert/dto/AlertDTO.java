package com.smhrd.web.farm.alert.dto;

import lombok.Data;


@Data
public class AlertDTO {   
    private Long anlsIdx;
    private String insectName;
//    private String ghArea;
    private String ghName;
    private Long anlsAcc;
    private String createdAt; // 초까지 나타내려면 String으로 해야된다 함
    private String notiCheck; // "N" or "Y"
    

    // gpt 내용
    private String userQes;
    private String gptContent;
    

}