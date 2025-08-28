package com.smhrd.web.camera;

import com.smhrd.web.ml.MlApiService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/camera")
@RequiredArgsConstructor
public class CameraController {

    private final MlApiService mlApiService;
    private final RestTemplate restTemplate;

    @PostMapping("/detect")
    public ResponseEntity<Map<String, Object>> detectInsect(@RequestBody Map<String, Object> frameData) {
        try {
            // ML 서버로 탐지 요청 전달
            String mlApiUrl = "http://localhost:8003/api/detect";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(frameData, headers);
            
            ResponseEntity<Map> response = restTemplate.postForEntity(mlApiUrl, request, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> detectionResult = response.getBody();
                
                // 탐지 결과가 있으면 후처리 (DB 저장, 알림 등)
                if (hasDetections(detectionResult)) {
                    processDetectionResult(detectionResult, frameData);
                }
                
                return ResponseEntity.ok(detectionResult);
            }
            
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "ML 서버 응답 실패"));
            
        } catch (Exception e) {
            log.error("해충 탐지 요청 처리 중 오류", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "서버 오류: " + e.getMessage()));
        }
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getCameraStats() {
        // 카메라 통계 조회 (구현 필요)
        Map<String, Object> stats = new HashMap<>();
        stats.put("message", "카메라 통계 조회 기능 구현 예정");
        return ResponseEntity.ok(stats);
    }

    private boolean hasDetections(Map<String, Object> result) {
        // 탐지 결과 확인 로직
        return result.containsKey("detections") && 
               result.get("detections") != null;
    }

    private void processDetectionResult(Map<String, Object> detectionResult, Map<String, Object> frameData) {
        try {
            // 탐지 결과 후처리
            log.info("탐지 결과 처리: {}", detectionResult);
            
            // 필요시 DB 저장, 알림 발송 등의 로직 추가
            
        } catch (Exception e) {
            log.error("탐지 결과 처리 중 오류", e);
        }
    }
}