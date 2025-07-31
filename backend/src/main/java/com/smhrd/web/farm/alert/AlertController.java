package com.smhrd.web.farm.alert;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.smhrd.web.farm.alert.dto.AlertDTO;
import com.smhrd.web.farm.alert.dto.AlertDetailDTO;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/user/alert")
public class AlertController {

    @Autowired
    private AlertService alertService;

    @Operation(summary = "알림 목록 조회")
    @GetMapping("/list/{farmIdx}")
    public ResponseEntity<List<AlertDTO>> getAlertList(@PathVariable Long farmIdx) {
        List<AlertDTO> alerts = alertService.getAlertList(farmIdx);
        return ResponseEntity.ok(alerts);
    }

    @Operation(summary = "알림 읽음 처리 후 상세 조회")
    @GetMapping("/read-and-detail/{anlsIdx}")
    public ResponseEntity<AlertDTO> readAndGetAlertDetail(@PathVariable Long anlsIdx) {
        alertService.markAsRead(anlsIdx); // 읽음 처리 먼저 하고
        AlertDTO alertDetail = alertService.getAlertDetail(anlsIdx); // 상세조회

        if (alertDetail != null) {
            return ResponseEntity.ok(alertDetail);
        } else {
            return ResponseEntity.notFound().build();
        }
    }
    
    @Operation(summary = "알림 눌렀을떄 해충 정보 상세 (구역확인, 동영상확인, GPT확인)")
    @GetMapping("/detail/{anlsIdx}")
    public ResponseEntity<AlertDetailDTO> getAlertDetail(@PathVariable Long anlsIdx) {
        AlertDetailDTO detail = alertService.getFullAlertDetail(anlsIdx);
        return ResponseEntity.ok(detail);
    }

}
