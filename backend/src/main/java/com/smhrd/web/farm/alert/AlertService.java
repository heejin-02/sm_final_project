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

    // 알림 목록 조회
    public List<AlertDTO> getAlertList(Long farmIdx) {
        return alertMapper.getAlertList(farmIdx);
    }
    
    // 알림 확인 처리
    public void markAsRead(Long anlsIdx) {
        alertMapper.updateNotiCheck(anlsIdx);
    }

    // 알림 상세 조회
    public AlertDTO getAlertDetail(Long anlsIdx) {
        return alertMapper.getAlertDetail(anlsIdx);
    }

    // 알림 상세 및 관련 정보 조회 (GreenhouseInfo, ImageList, GPT 결과, 피드백 포함)
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

    // 피드백 조회
    public FeedbackDTO getFeedback(Long anlsIdx) {
        return alertMapper.getFeedback(anlsIdx);
    }

    // 피드백 등록
    public void insertFeedback(FeedbackDTO feedback) {
        alertMapper.insertFeedback(feedback);
    }

    // 피드백 수정
    public void updateFeedback(Long anlsIdx, FeedbackDTO feedback) {
        alertMapper.updateFeedback(feedback);
    }
}
