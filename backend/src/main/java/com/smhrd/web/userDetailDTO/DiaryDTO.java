package com.smhrd.web.userDetailDTO;

import lombok.Data;

@Data
public class DiaryDTO {
    private int diaryIdx;
    private int farmIdx;
    private String title;
    private String content;
    private String regDate;
}
