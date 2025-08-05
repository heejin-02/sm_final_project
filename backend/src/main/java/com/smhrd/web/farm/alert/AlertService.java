package com.smhrd.web.farm.alert;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.smhrd.web.farm.alert.dto.AlertDTO;
import com.smhrd.web.farm.alert.dto.AlertDetailDTO;
import com.smhrd.web.farm.alert.dto.FeedbackDTO;
import com.smhrd.web.farm.alert.dto.GptResultDTO;
import com.smhrd.web.farm.alert.dto.GreenhouseInfoDTO;
import com.smhrd.web.farm.alert.dto.ImageInfoDTO;

@Service
public class AlertService {

    @Autowired
    private AlertMapper alertMapper;

    public List<AlertDTO> getAlertList(Long farmIdx) {
        return alertMapper.getAlertList(farmIdx);
    }
    
    public void markAsRead(Long anlsIdx) {
        System.out.println("updateNotiCheck 실행, anlsIdx = " + anlsIdx);
        alertMapper.updateNotiCheck(anlsIdx);
    }

    public AlertDTO getAlertDetail(Long anlsIdx) {
        return alertMapper.getAlertDetail(anlsIdx);
    }
    
    public AlertDetailDTO getFullAlertDetail(Long anlsIdx) {
        GreenhouseInfoDTO ghInfo = alertMapper.getGreenhouseInfo(anlsIdx);
        List<ImageInfoDTO> images = alertMapper.getImageList(anlsIdx);
        GptResultDTO gpt = alertMapper.getGptResult(anlsIdx);
        FeedbackDTO feedback = alertMapper.getFeedback(anlsIdx);

        AlertDetailDTO dto = new AlertDetailDTO();
        dto.setGreenhouseInfo(ghInfo);
        dto.setImageList(images);
        dto.setGptResult(gpt);
        dto.setFeedback(feedback);
        return dto;
    }
    
    public void insertFeedback(FeedbackDTO feedback) {
        alertMapper.insertFeedback(feedback);
    }




}
