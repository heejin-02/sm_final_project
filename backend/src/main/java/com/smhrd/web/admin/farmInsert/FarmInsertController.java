package com.smhrd.web.admin.farmInsert;

import java.io.File;
import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@Tag(name = "Farm Insert Controller", description = "농장 추가 API")
@RestController
@RequestMapping("/api/farm")
public class FarmInsertController {

    @Autowired
    private FarmInsertService farmInsertService;

	@Operation(summary = "농장을 추가합니다.")
    @PostMapping(value = "/insertFarm", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> insertFarm(@ModelAttribute FarmInsertDTO dto) {
        MultipartFile file = dto.getFarmImg();
        
        if (file != null && !file.isEmpty()) {
            String fileName = file.getOriginalFilename();
            String uploadPath = "C:/upload/farm/";

            File saveFile = new File(uploadPath + fileName);

            try {
                file.transferTo(saveFile);
                dto.setFarmImgName(fileName);
            } catch (IOException e) {
                e.printStackTrace();
                return ResponseEntity.badRequest().body("파일 저장 실패");
            }
        }

        boolean result = farmInsertService.insertFarm(dto);
        if (result) {
            return ResponseEntity.ok("농장 등록 성공");
        } else {
            return ResponseEntity.status(500).body("농장 등록 실패");
        }
    }
}
