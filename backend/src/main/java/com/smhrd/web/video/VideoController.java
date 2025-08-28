package com.smhrd.web.video;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/video")
public class VideoController {

    @Value("${file.upload.dir}")
    private String uploadDir;

    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadVideo(
            @RequestParam("video") MultipartFile videoFile,
            @RequestParam("camera_id") String cameraId,
            @RequestParam("gh_idx") Long ghIdx,
            @RequestParam("detection_count") int detectionCount,
            @RequestParam("recording_start_time") double recordingStartTime,
            @RequestParam("frame_count") int frameCount) {
        
        Map<String, Object> response = new HashMap<>();
        
        try {
            // 날짜별 디렉토리 생성
            String dateFolder = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            Path dateDirPath = Paths.get(uploadDir, dateFolder);
            Files.createDirectories(dateDirPath);
            
            // 파일명 생성
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("HHmmss"));
            String fileName = String.format("detection_%s_%s_%d.mp4", cameraId, timestamp, detectionCount);
            Path filePath = dateDirPath.resolve(fileName);
            
            // 비디오 파일 저장
            videoFile.transferTo(filePath.toFile());
            
            // 상대 경로 생성 (프론트엔드에서 사용)
            String relativePath = dateFolder + "/" + fileName;
            
            log.info("🎥 비디오 업로드 성공: {} ({}MB, {}개 탐지)", 
                    relativePath, videoFile.getSize() / (1024 * 1024), detectionCount);
            
            response.put("success", true);
            response.put("message", "비디오 업로드 성공");
            response.put("video_path", relativePath);
            response.put("file_size", videoFile.getSize());
            response.put("detection_count", detectionCount);
            response.put("frame_count", frameCount);
            
            // TODO: 데이터베이스에 비디오 정보 저장
            // insertVideoRecord(cameraId, ghIdx, relativePath, detectionCount, frameCount);
            
            return ResponseEntity.ok(response);
            
        } catch (IOException e) {
            log.error("❌ 비디오 업로드 실패: {}", e.getMessage());
            response.put("success", false);
            response.put("message", "비디오 업로드 실패: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    @GetMapping("/stream/{dateFolder}/{fileName}")
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