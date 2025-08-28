package com.smhrd.web.ml;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MlApiService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private MlMapper mlMapper;

    @Value("${ml.api.base-url}")
    private String mlApiBaseUrl;

    private HttpHeaders createHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return headers;
    }

    /**
     * 일간 GPT 요약 조회
     */
    public Map<String, Object> getDailyGptSummary(Long farmIdx, String date) {
        String url = UriComponentsBuilder.fromHttpUrl(mlApiBaseUrl + "/api/daily-gpt-summary")
                .queryParam("farm_idx", farmIdx)
                .queryParam("date", date)
                .toUriString();

        try {
            System.out.println("[DEBUG] ML API 호출: " + url);
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            System.out.println("[DEBUG] ML API 응답 상태: " + response.getStatusCode());
            System.out.println("[DEBUG] ML API 응답 데이터: " + response.getBody());
            return response.getBody();
        } catch (Exception e) {
            System.err.println("[ERROR] ML API 호출 실패: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * 월간 GPT 요약 조회
     */
    public Map<String, Object> getMonthlyGptSummary(Long farmIdx, String month) {
        String url = UriComponentsBuilder.fromHttpUrl(mlApiBaseUrl + "/api/monthly-gpt-summary")
                .queryParam("farm_idx", farmIdx)
                .queryParam("month", month)
                .toUriString();

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * 연간 GPT 요약 조회
     */
    public Map<String, Object> getYearlyGptSummary(Long farmIdx, String year) {
        String url = UriComponentsBuilder.fromHttpUrl(mlApiBaseUrl + "/api/yearly-gpt-summary")
                .queryParam("farm_idx", farmIdx)
                .queryParam("year", year)
                .toUriString();

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * 이미지 인덱스 기반 해충 요약
     */
    public Map<String, Object> getSummaryByImgIdx(Long imgIdx) {
        String url = UriComponentsBuilder.fromHttpUrl(mlApiBaseUrl + "/api/summary-by-imgidx")
                .queryParam("imgIdx", imgIdx)
                .toUriString();

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * 해충 방제 정보 요약
     */
    public Map<String, Object> getInsectSummary(String insectName) {
        String url = mlApiBaseUrl + "/api/insect-summary";
        
        HttpHeaders headers = createHeaders();
        Map<String, String> requestBody = Map.of("insect_name", insectName);
        HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * RAG 질문 응답
     */
    public Map<String, Object> askQuestion(String question) {
        String url = mlApiBaseUrl + "/api/ask";
        
        HttpHeaders headers = createHeaders();
        Map<String, String> requestBody = Map.of("question", question);
        HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("ML API 호출 실패: " + e.getMessage(), e);
        }
    }

    /**
     * 온실 ID로 전화번호 조회
     */
    public Map<String, Object> getUserPhone(Long ghIdx) {
        String url = UriComponentsBuilder.fromHttpUrl(mlApiBaseUrl + "/api/get-phone")
                .queryParam("gh_idx", ghIdx)
                .toUriString();

        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("전화번호 조회 실패: " + e.getMessage(), e);
        }
    }

    /**
     * ML API 서버 상태 확인
     */
    public boolean checkMlApiHealth() {
        try {
            String url = mlApiBaseUrl + "/";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            return response.getStatusCode() == HttpStatus.OK;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 농장 이름 조회 - ML API 지원용
     */
    public Map<String, Object> getFarmName(Long farmIdx) {
        return mlMapper.selectFarmName(farmIdx);
    }

    // ========================================
    // Oracle DB 직접 연결 대체 API들
    // ========================================

    /**
     * 해충 탐지 집계 데이터 조회 (get_aggregated_analysis_text 대체)
     */
    public Map<String, Object> getAggregatedAnalysisText(String insectName) {
        List<Map<String, Object>> detectionData = mlMapper.getAggregatedAnalysisData(insectName);
        
        if (detectionData == null || detectionData.isEmpty()) {
            return Map.of(
                "summary", "최근 3일간 탐지된 이력이 없습니다.",
                "mostLocation", "",
                "insectName", insectName
            );
        }

        // 위치별 탐지 횟수 계산
        Map<String, Integer> locationCount = new HashMap<>();
        double totalConfidence = 0;
        
        for (Map<String, Object> data : detectionData) {
            String ghName = (String) data.get("ghName");
            Number confidence = (Number) data.get("confidence");
            
            locationCount.put(ghName, locationCount.getOrDefault(ghName, 0) + 1);
            if (confidence != null) {
                totalConfidence += confidence.doubleValue();
            }
        }

        // 가장 많이 탐지된 위치 찾기
        String mostLocation = locationCount.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("");
        
        int mostLocationCount = locationCount.getOrDefault(mostLocation, 0);
        double avgConfidence = totalConfidence / detectionData.size();

        String summary = String.format(
            "최근 3일간 '%s'는 총 %d회 탐지되었습니다. " +
            "그 중 '%s' 위치에서 %d회 감지되었고, " +
            "평균 신뢰도는 %.1f%%입니다.",
            insectName, detectionData.size(), mostLocation, mostLocationCount, avgConfidence
        );

        return Map.of(
            "summary", summary,
            "mostLocation", mostLocation,
            "insectName", insectName
        );
    }

    /**
     * GPT 요약 저장 (insert_gpt_summary 대체)
     */
    public void insertGptSummary(Long anlsIdx, String userQes, String gptContent) {
        try {
            mlMapper.insertGptSummary(anlsIdx, userQes, gptContent);
            System.out.println("[DB] GPT 응답 저장 완료");
        } catch (Exception e) {
            System.err.println("[DB ERROR] GPT 응답 저장 실패: " + e.getMessage());
            throw new RuntimeException("GPT 응답 저장 실패", e);
        }
    }

    /**
     * 이미지 인덱스로 해충 정보 조회 (get_summary_by_imgidx 관련)
     */
    public Map<String, Object> getInsectInfoByImgIdx(Long imgIdx) {
        Map<String, Object> result = mlMapper.getInsectInfoByImgIdx(imgIdx);
        if (result == null || result.isEmpty()) {
            throw new RuntimeException("해당 IMG_IDX에 대한 해충 정보가 없습니다.");
        }
        return result;
    }

    /**
     * 온실 ID로 사용자 전화번호 조회 (get_user_phone 대체)
     */
    public Map<String, Object> getUserPhoneByGhIdx(Long ghIdx) {
        Map<String, Object> result = mlMapper.getUserPhoneByGhIdx(ghIdx);
        if (result == null || result.get("userPhone") == null) {
            throw new RuntimeException("전화번호를 찾을 수 없습니다.");
        }
        return Map.of("phone", result.get("userPhone"));
    }

    /**
     * 오늘 탐지 요약 조회 (get_today_detection_summary 대체)
     */
    public List<Map<String, Object>> getTodayDetectionSummary() {
        return mlMapper.getTodayDetectionSummary();
    }

    /**
     * 대시보드 요약 업서트 (upsert_dashboard_summary 대체)
     */
    public void upsertDashboardSummary(Long anlsIdx, String gptContent) {
        try {
            Map<String, Object> existing = mlMapper.getExistingDashboardSummary();
            
            if (existing != null && existing.get("gptIdx") != null) {
                Long gptIdx = ((Number) existing.get("gptIdx")).longValue();
                mlMapper.updateDashboardSummary(gptIdx, gptContent);
            } else {
                mlMapper.insertDashboardSummary(gptContent, anlsIdx);
            }
            System.out.println("[DB] 대시보드 요약 저장 완료");
        } catch (Exception e) {
            System.err.println("[DB ERROR] 대시보드 요약 업서트 실패: " + e.getMessage());
            throw new RuntimeException("대시보드 요약 저장 실패", e);
        }
    }

    /**
     * Twilio 전화 발신용 데이터 조회 (twilio_call 대체)
     */
    public Map<String, Object> getTwilioCallData() {
        Map<String, Object> result = mlMapper.getTwilioCallData();
        if (result == null || result.isEmpty()) {
            return Map.of("message", "최근 탐지된 해충 정보가 없습니다.");
        }
        
        String ghName = (String) result.get("ghName");
        String insectName = (String) result.get("insectName");
        String message = String.format("%s에서 %s가 탐지되었습니다. 확인해 주세요.", ghName, insectName);
        
        return Map.of("message", message);
    }
}