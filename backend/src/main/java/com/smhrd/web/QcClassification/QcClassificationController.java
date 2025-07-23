package com.smhrd.web.QcClassification;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/qc-classification")
@RequiredArgsConstructor
public class QcClassificationController {

    private final QcClassificationService service;

    @PostMapping
    public ResponseEntity<String> receiveDetection(@RequestBody QcClassificationDTO dto) {
        service.saveClassification(dto);
        return ResponseEntity.ok("저장 완료");
    }
}

