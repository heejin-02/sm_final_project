package com.smhrd.web.ml;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/ml")
@Tag(name = "ML API Controller", description = "ML 서버 프록시 API")
@CrossOrigin(origins = {"http://localhost:5173", "http://192.168.219.49:5173"}, allowCredentials = "true")
public class MlApiController {

    @Autowired
    private MlApiService mlApiService;

    @GetMapping("/daily-gpt-summary")
    @Operation(summary = "일간 GPT 요약 조회")
    public ResponseEntity<Map<String, Object>> getDailyGptSummary(
            @RequestParam("farm_idx") Long farmIdx,
            @RequestParam String date) {
        try {
            Map<String, Object> result = mlApiService.getDailyGptSummary(farmIdx, date);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/monthly-gpt-summary")
    @Operation(summary = "월간 GPT 요약 조회")
    public ResponseEntity<Map<String, Object>> getMonthlyGptSummary(
            @RequestParam("farm_idx") Long farmIdx,
            @RequestParam String month) {
        try {
            Map<String, Object> result = mlApiService.getMonthlyGptSummary(farmIdx, month);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/yearly-gpt-summary")
    @Operation(summary = "연간 GPT 요약 조회")
    public ResponseEntity<Map<String, Object>> getYearlyGptSummary(
            @RequestParam("farm_idx") Long farmIdx,
            @RequestParam String year) {
        try {
            Map<String, Object> result = mlApiService.getYearlyGptSummary(farmIdx, year);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/summary-by-imgidx")
    @Operation(summary = "이미지 인덱스 기반 해충 요약")
    public ResponseEntity<Map<String, Object>> getSummaryByImgIdx(
            @RequestParam("imgIdx") Long imgIdx) {
        try {
            Map<String, Object> result = mlApiService.getSummaryByImgIdx(imgIdx);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/insect-summary")
    @Operation(summary = "해충 방제 정보 요약")
    public ResponseEntity<Map<String, Object>> getInsectSummary(
            @RequestBody Map<String, String> request) {
        try {
            String insectName = request.get("insect_name");
            Map<String, Object> result = mlApiService.getInsectSummary(insectName);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/ask")
    @Operation(summary = "RAG 질문 응답")
    public ResponseEntity<Map<String, Object>> askQuestion(
            @RequestBody Map<String, String> request) {
        try {
            String question = request.get("question");
            Map<String, Object> result = mlApiService.askQuestion(question);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/get-phone")
    @Operation(summary = "온실 ID로 전화번호 조회")
    public ResponseEntity<Map<String, Object>> getUserPhone(
            @RequestParam("gh_idx") Long ghIdx) {
        try {
            Map<String, Object> result = mlApiService.getUserPhone(ghIdx);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(404).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/farm-name")
    @Operation(summary = "농장 인덱스로 농장 이름 조회")
    public ResponseEntity<Map<String, Object>> getFarmName(
            @RequestParam("farm_idx") Long farmIdx) {
        try {
            Map<String, Object> result = mlApiService.getFarmName(farmIdx);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(404).body(Map.of("error", e.getMessage()));
        }
    }

    // ========================================
    // Oracle DB 직접 연결 대체 API들
    // ========================================

    @GetMapping("/aggregated-analysis-text")
    @Operation(summary = "해충 탐지 집계 데이터 조회")
    public ResponseEntity<Map<String, Object>> getAggregatedAnalysisText(
            @RequestParam("insect_name") String insectName) {
        try {
            Map<String, Object> result = mlApiService.getAggregatedAnalysisText(insectName);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/gpt-summary")
    @Operation(summary = "GPT 요약 저장")
    public ResponseEntity<Map<String, Object>> insertGptSummary(
            @RequestBody Map<String, Object> request) {
        try {
            Long anlsIdx = ((Number) request.get("anls_idx")).longValue();
            String userQes = (String) request.get("user_qes");
            String gptContent = (String) request.get("gpt_content");
            
            mlApiService.insertGptSummary(anlsIdx, userQes, gptContent);
            return ResponseEntity.ok(Map.of("message", "GPT 응답 저장 완료"));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/insect-info-by-imgidx")
    @Operation(summary = "이미지 인덱스로 해충 정보 조회")
    public ResponseEntity<Map<String, Object>> getInsectInfoByImgIdx(
            @RequestParam("img_idx") Long imgIdx) {
        try {
            Map<String, Object> result = mlApiService.getInsectInfoByImgIdx(imgIdx);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(404).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/user-phone-by-ghidx")
    @Operation(summary = "온실 ID로 사용자 전화번호 조회")
    public ResponseEntity<Map<String, Object>> getUserPhoneByGhIdx(
            @RequestParam("gh_idx") Long ghIdx) {
        System.out.println("[API] 전화번호 조회 요청 받음 - GH_IDX: " + ghIdx);
        
        try {
            Map<String, Object> result = mlApiService.getUserPhoneByGhIdx(ghIdx);
            System.out.println("[API] ✅ 전화번호 조회 성공 응답 - GH_IDX: " + ghIdx);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            System.out.println("[API] ❌ 전화번호 조회 실패 - GH_IDX: " + ghIdx + ", 오류: " + e.getMessage());
            return ResponseEntity.status(404).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/today-detection-summary")
    @Operation(summary = "오늘 탐지 요약 조회")
    public ResponseEntity<List<Map<String, Object>>> getTodayDetectionSummary() {
        try {
            List<Map<String, Object>> result = mlApiService.getTodayDetectionSummary();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(List.of(Map.of("error", e.getMessage())));
        }
    }

    @PostMapping("/dashboard-summary")
    @Operation(summary = "대시보드 요약 업서트")
    public ResponseEntity<Map<String, Object>> upsertDashboardSummary(
            @RequestBody Map<String, Object> request) {
        try {
            Long anlsIdx = ((Number) request.get("anls_idx")).longValue();
            String gptContent = (String) request.get("gpt_content");
            
            mlApiService.upsertDashboardSummary(anlsIdx, gptContent);
            return ResponseEntity.ok(Map.of("message", "대시보드 요약 저장 완료"));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/signalwire-call-data")
    @Operation(summary = "SignalWire 전화 발신용 데이터 조회")
    public ResponseEntity<Map<String, Object>> getSignalWireCallData() {
        try {
            Map<String, Object> result = mlApiService.getSignalWireCallData();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }
}