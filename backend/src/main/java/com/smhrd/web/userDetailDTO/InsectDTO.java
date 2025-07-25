package com.smhrd.web.userDetailDTO;

import lombok.Data;

@Data
public class InsectDTO {
    private int insectIdx;
    private int farmIdx;
    private String insectType;
    private int insectCount;
    private String detectedDate;

}
