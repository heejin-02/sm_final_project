package com.smhrd.web.QcImage;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class QcImageService {

    @Autowired
    private QcImageMapper mapper;

    public int insertImage(QcImageDTO dto) {
        return mapper.insertImage(dto);
    }

    public int getLastInsertedIdx() {
        return mapper.getLastInsertedIdx();
    }
}
