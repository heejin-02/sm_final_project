package com.smhrd.web.QcImage;

import lombok.Data;

@Data
public class QcImageDTO {
    private Long imgIdx;
    private String imgName;
    private Long imgSize;
    private String imgExt;
    private String createdAt;
    private Double imgX;
    private Double imgY;
    private String imgUrl;
    private Long ghIdx;
}
