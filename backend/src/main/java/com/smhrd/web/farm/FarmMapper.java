package com.smhrd.web.farm;

import org.apache.ibatis.annotations.Mapper;


@Mapper
public interface FarmMapper {
    public FarmDetailResponseDTO selectFarmDetail(Long farmIdx);
}
