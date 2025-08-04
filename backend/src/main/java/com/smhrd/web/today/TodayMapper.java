package com.smhrd.web.today;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TodayMapper {
    TodayResultDTO getTodayResult();
}
