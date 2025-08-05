package com.smhrd.web.today;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class TodayService {

    @Autowired
    private TodayMapper todayMapper;

    public TodayResultDTO getTodayResultByFarm(Long farmIdx) {
        return todayMapper.getTodayResultByFarm(farmIdx);
    }
    
    public List<TodayGreenhouseDTO> getTodayResultByGreenhouse(Long farmIdx) {
        return todayMapper.getTodayResultByGreenhouse(farmIdx);
    }

    public List<String> getTodayGptReportsByFarm(Long farmIdx) {
        return todayMapper.getTodayGptReportsByFarm(farmIdx);
    }



}
