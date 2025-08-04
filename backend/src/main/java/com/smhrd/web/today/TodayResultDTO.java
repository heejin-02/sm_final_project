package com.smhrd.web.today;

import lombok.Data;

@Data
public class TodayResultDTO {
    private String todayDate;
    private int todayCount;    
    private int insectTypeCount;
    private int zoneCount;
}
