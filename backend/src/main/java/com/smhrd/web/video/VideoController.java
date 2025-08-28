package com.smhrd.web.video;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
@RequestMapping("/videos")
public class VideoController {

    @Value("${file.upload.dir}")
    private String uploadDir;

    @GetMapping("/{dateFolder}/{fileName}")
    public ResponseEntity<Resource> getVideo(
            @PathVariable String dateFolder,
            @PathVariable String fileName) {
        
        try {
            // 파일 경로 생성
            Path filePath = Paths.get(uploadDir, dateFolder, fileName);
            File file = filePath.toFile();
            
            if (!file.exists()) {
                return ResponseEntity.notFound().build();
            }
            
            // 파일 리소스 생성
            Resource resource = new FileSystemResource(file);
            
            // Content-Type 설정
            String contentType = "video/mp4";
            
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + fileName + "\"")
                    .body(resource);
                    
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}