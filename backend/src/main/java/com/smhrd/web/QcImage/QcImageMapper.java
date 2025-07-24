package com.smhrd.web.QcImage;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface QcImageMapper {
    public int insertImage(QcImageDTO dto);
    public int getLastInsertedIdx(); // 또는 Long 반환해도 무관
}
