package com.smhrd.web.ml;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;
import java.util.Map;

@Mapper
public interface MlMapper {
    Map<String, Object> selectFarmName(Long farmIdx);
    
    // GPT 요약 관련
    List<Map<String, Object>> getAggregatedAnalysisData(@Param("insectName") String insectName);
    void insertGptSummary(@Param("anlsIdx") Long anlsIdx, @Param("userQes") String userQes, @Param("gptContent") String gptContent);
    
    // 기본 기능
    Map<String, Object> getInsectInfoByImgIdx(@Param("imgIdx") Long imgIdx);
    Map<String, Object> getUserPhoneByGhIdx(@Param("ghIdx") Long ghIdx);
    
    // 부가 기능  
    List<Map<String, Object>> getTodayDetectionSummary();
    Map<String, Object> getExistingDashboardSummary();
    void updateDashboardSummary(@Param("gptIdx") Long gptIdx, @Param("gptContent") String gptContent);
    void insertDashboardSummary(@Param("gptContent") String gptContent, @Param("anlsIdx") Long anlsIdx);
    Map<String, Object> getSignalWireCallData();
}