// FarmDTO.java
package com.smhrd.web.userDetailDTO;

import java.time.LocalDateTime;
import lombok.Data;

@Data
public class FarmDetailDTO {
    private Long farmIdx;
    private String farmName;
    private String farmAddr;
    private String farmPhone;
    private String farmCrops;
    private String farmArea;
    private String farmImg;
    private LocalDateTime createdAt;
}
