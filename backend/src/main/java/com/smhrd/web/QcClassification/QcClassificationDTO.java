package com.smhrd.web.QcClassification;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class QcClassificationDTO {
	
    private String anlsModel;
    private String anlsContent;
    private String anlsResult;
    private String createdAt;
    private int insectIdx;
    private int imgIdx;
    private String notiCheck;
    private int ghIdx;
    private int anlsAcc;
}

