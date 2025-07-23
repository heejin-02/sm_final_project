package com.smhrd.web.QcImage;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/qc-images")
public class QcImageController {

    @Autowired
    private QcImageService service;

    @PostMapping
    public ResponseEntity<Integer> saveImage(@RequestBody QcImageDTO dto) {
        int result = service.insertImage(dto);

        if (result > 0) {
            int imgIdx = service.getLastInsertedIdx();
            return ResponseEntity.ok(imgIdx);
        } else {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(-1);
        }
    }
}
