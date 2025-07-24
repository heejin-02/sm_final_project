package com.smhrd.web.QcClassification;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface QcClassificationMapper {
    public void insertClassification(QcClassificationDTO dto);
}
