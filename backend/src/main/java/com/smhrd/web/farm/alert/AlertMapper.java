package com.smhrd.web.farm.alert;

import java.util.List;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import com.smhrd.web.farm.alert.dto.AlertDTO;
import com.smhrd.web.farm.alert.dto.GptResultDTO;
import com.smhrd.web.farm.alert.dto.GreenhouseInfoDTO;
import com.smhrd.web.farm.alert.dto.ImageInfoDTO;

@Mapper
public interface AlertMapper {

    // 알림 목록 조회
    public List<AlertDTO> getAlertList(@Param("farmIdx") Long farmIdx);

    // 알림 확인 처리
    public int updateNotiCheck(@Param("anlsIdx") Long anlsIdx);


    // 알림 세부사항, gpt api 알림
    AlertDTO getAlertDetail(Long anlsIdx);

    GreenhouseInfoDTO getGreenhouseInfo(Long anlsIdx);
    List<ImageInfoDTO> getImageList(Long anlsIdx);
    GptResultDTO getGptResult(Long anlsIdx);

}

