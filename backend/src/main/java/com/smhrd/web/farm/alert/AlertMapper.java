package com.smhrd.web.farm.alert;

import java.util.List;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import com.smhrd.web.farm.alert.dto.AlertDTO;
import com.smhrd.web.farm.alert.dto.FeedbackDTO;
import com.smhrd.web.farm.alert.dto.GptResultDTO;
import com.smhrd.web.farm.alert.dto.GreenhouseInfoDTO;
import com.smhrd.web.farm.alert.dto.ImageInfoDTO;

@Mapper
public interface AlertMapper {

    // 알림 목록 조회
    List<AlertDTO> getAlertList(@Param("farmIdx") Long farmIdx);

    // 알림 읽음 처리
    int updateNotiCheck(@Param("anlsIdx") Long anlsIdx);

    // 알림 세부사항 조회
    AlertDTO getAlertDetail(Long anlsIdx);

    // 온실 정보 조회
    GreenhouseInfoDTO getGreenhouseInfo(Long anlsIdx);

    // 이미지 목록 조회
    List<ImageInfoDTO> getImageList(Long anlsIdx);

    // GPT 결과 조회
    GptResultDTO getGptResult(Long anlsIdx);

    // 피드백 등록
    int insertFeedback(FeedbackDTO feedback);

    // 피드백 조회
    FeedbackDTO getFeedback(Long anlsIdx);

    // 피드백 수정
    int updateFeedback(FeedbackDTO feedback);
}
