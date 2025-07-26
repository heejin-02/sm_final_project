package com.smhrd.web.QcImage;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class QcImageService {

    @Autowired
    private QcImageMapper mapper;

    public int insertImage(QcImageDTO dto) {
    	System.out.println("[LOG] QcImageService.insertImage() 호출됨");
        System.out.println("[LOG] 이미지 이름: " + dto.getImgName());
        return mapper.insertImage(dto);
    }

    public int getLastInsertedIdx() {
        return mapper.getLastInsertedIdx();
    }
}
