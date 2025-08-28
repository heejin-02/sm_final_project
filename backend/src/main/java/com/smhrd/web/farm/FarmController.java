package com.smhrd.web.farm;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;


import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api/farms")
@Tag(name = "Farm Controller", description = "농장조회 API")
public class FarmController {

    @Autowired
    private FarmService farmService;

    @GetMapping("/{farmIdx}/detail")
    @Operation(summary = "farmIdx로 선택된 농장의 상세 정보를 조회합니다.")
    public ResponseEntity<FarmDetailResponseDTO> getFarmDetail(@PathVariable Long farmIdx) {
        FarmDetailResponseDTO detail = farmService.getFarmDetail(farmIdx);
        return ResponseEntity.ok(detail);
    }
}
