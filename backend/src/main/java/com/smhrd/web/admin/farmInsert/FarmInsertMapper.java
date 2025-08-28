package com.smhrd.web.admin.farmInsert;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface FarmInsertMapper {
    int insertFarm(FarmInsertDTO dto);
}
