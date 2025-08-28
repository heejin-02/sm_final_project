package com.smhrd.web.farm.alert;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.smhrd.web.farm.alert.dto.AlertDTO;
import com.smhrd.web.farm.alert.dto.AlertDetailDTO;
import com.smhrd.web.farm.alert.dto.FeedbackDTO;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/user/alert")
@Tag(name = "Alert Controller", description = "농장 알림 관련 API")
public class AlertController {

    @Autowired
    private AlertService alertService;

    @GetMapping("/list/{farmIdx}")
    @Operation(summary = "알림 목록을 조회합니다.")
    public ResponseEntity<List<AlertDTO>> getAlertList(@PathVariable Long farmIdx) {
        List<AlertDTO> alerts = alertService.getAlertList(farmIdx);
        return ResponseEntity.ok(alerts);
    }

    @GetMapping("/read-and-detail/{anlsIdx}")
    @Operation(summary = "알림 읽음 처리 후 상세 조회합니다.")
    public ResponseEntity<AlertDTO> readAndGetAlertDetail(@PathVariable Long anlsIdx) {
        alertService.markAsRead(anlsIdx); // 읽음 처리 먼저 하고
        AlertDTO alertDetail = alertService.getAlertDetail(anlsIdx); // 상세조회

        if (alertDetail != null) {
            return ResponseEntity.ok(alertDetail);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
    
    @GetMapping("/detail/{anlsIdx}")
    @Operation(summary = "알림 클릭시 해충 정보 상세보기 (구역확인, 동영상확인, GPT확인)")
    public ResponseEntity<AlertDetailDTO> getAlertDetail(@PathVariable Long anlsIdx) {
        AlertDetailDTO detail = alertService.getFullAlertDetail(anlsIdx);
        return ResponseEntity.ok(detail);
    }
    
    @PostMapping("/feedback")
    @Operation(summary = "피드백 등록합니다.")
    public ResponseEntity<String> insertFeedback(@RequestBody FeedbackDTO feedback) {
        alertService.insertFeedback(feedback);
        return ResponseEntity.ok("피드백이 성공적으로 등록되었습니다.");
    }

    // Controller
    @GetMapping("/feedback/{anlsIdx}")
    @Operation(summary = "피드백 조회")
    public ResponseEntity<FeedbackDTO> getFeedback(@PathVariable Long anlsIdx) {
        FeedbackDTO feedback = alertService.getFeedback(anlsIdx);
        return ResponseEntity.ok(feedback);
    }

    // Controller
    @PutMapping("/feedback/{anlsIdx}")
    @Operation(summary = "피드백 수정")
    public ResponseEntity<String> updateFeedback(@PathVariable Long anlsIdx, @RequestBody FeedbackDTO feedback) {
        alertService.updateFeedback(anlsIdx, feedback);
        return ResponseEntity.ok("피드백이 성공적으로 수정되었습니다.");
    }

}
