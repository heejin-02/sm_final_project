package com.smhrd.web.today;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TodayMapper {
	TodayResultDTO getTodayResultByFarm(Long farmIdx);	
	List<TodayGreenhouseDTO> getTodayResultByGreenhouse(Long farmIdx);
	List<String> getTodayGptReportsByFarm(Long farmIdx);

}
