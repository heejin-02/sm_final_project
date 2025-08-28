package com.smhrd.web.today;

import lombok.Data;

@Data
public class TodayGreenhouseDTO {
    private Long ghIdx;
    private String ghName;
    private int todayInsectCount;
}
