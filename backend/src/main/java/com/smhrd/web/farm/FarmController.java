package com.smhrd.web.farm;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;


import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/api/farms")
public class FarmController {

    @Autowired
    private FarmService farmService;

    @Operation(summary = "farmIdx로 농장 상세 정보 조회", description = "선택된 농장의 상세 정보를 조회합니다.")
    @GetMapping("/{farmIdx}/detail")
    public ResponseEntity<FarmDetailResponseDTO> getFarmDetail(@PathVariable Long farmIdx) {
        FarmDetailResponseDTO detail = farmService.getFarmDetail(farmIdx);
        return ResponseEntity.ok(detail);
    }
}
