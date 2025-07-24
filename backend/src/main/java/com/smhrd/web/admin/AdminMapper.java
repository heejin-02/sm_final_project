package com.smhrd.web.admin;

import java.util.List;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AdminMapper {
    List<AdminDTO> selectFarmList();
}
