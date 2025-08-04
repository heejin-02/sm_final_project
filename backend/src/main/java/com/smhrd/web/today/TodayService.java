package com.smhrd.web.today;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class TodayService {

    @Autowired
    private TodayMapper todayMapper;

    public TodayResultDTO getTodayResult() {
        return todayMapper.getTodayResult();
    }
}
