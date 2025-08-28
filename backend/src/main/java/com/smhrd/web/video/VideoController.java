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

import com.smhrd.web.QcImage.QcImageDTO;
import com.smhrd.web.QcImage.QcImageMapper;

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
public class VideoController {

    @Value("${file.upload.dir}")
    private String uploadDir;
    
    private final QcImageMapper imageMapper;
    
    public VideoController(QcImageMapper imageMapper) {
        this.imageMapper = imageMapper;
    }

    @PostMapping("/api/video/upload")
    public ResponseEntity<Map<String, Object>> uploadVideo(
            @RequestParam("video") MultipartFile videoFile,
            @RequestParam("camera_id") String cameraId,
            @RequestParam("gh_idx") Long ghIdx,
            @RequestParam("detection_count") int detectionCount,
            @RequestParam("recording_start_time") double recordingStartTime,
            @RequestParam("frame_count") int frameCount) {
        
        Map<String, Object> response = new HashMap<>();
        
        try {
            log.info("📁 현재 작업 디렉토리: {}", System.getProperty("user.dir"));
            log.info("📁 설정된 업로드 디렉토리: {}", uploadDir);
            
            // 업로드 디렉토리가 상대경로인 경우 현재 디렉토리 기준으로 생성
            Path uploadPath = Paths.get(uploadDir);
            if (!uploadPath.isAbsolute()) {
                uploadPath = Paths.get(System.getProperty("user.dir"), uploadDir);
            }
            
            // 날짜별 디렉토리 생성
            String dateFolder = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            Path dateDirPath = uploadPath.resolve(dateFolder);
            Files.createDirectories(dateDirPath);
            
            log.info("📁 비디오 저장 경로: {}", dateDirPath.toAbsolutePath());
            
            // 파일명 생성 (기존 QC_IMAGES 형식과 통일)
            String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("HHmmss"));
            // classId 대신 해충 종류별 인덱스 사용 (탐지된 해충이 없으면 0)
            int classId = detectionCount > 0 ? 1 : 0;  // 임시로 1 사용, 나중에 주요 해충 종류로 변경 가능
            String fileName = String.format("%d_%s_%s.mp4", classId, dateFolder, time);
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
            
            // 데이터베이스에 비디오 정보 저장 (QC_IMAGES 테이블)
            Long imgIdx = saveVideoToDatabase(fileName, relativePath, videoFile, ghIdx);
            
            response.put("img_idx", imgIdx);
            
            return ResponseEntity.ok(response);
            
        } catch (IOException e) {
            log.error("❌ 비디오 업로드 실패: {}", e.getMessage());
            response.put("success", false);
            response.put("message", "비디오 업로드 실패: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    private Long saveVideoToDatabase(String fileName, String relativePath, MultipartFile videoFile, Long ghIdx) {
        try {
            // 비디오 URL 생성
            String serverIp = java.net.InetAddress.getLocalHost().getHostAddress();
            // 기존 프론트엔드 호환성을 위해 /videos/ 경로 사용
            String videoUrl = "http://" + serverIp + ":8095/videos/" + relativePath;
            
            // QC_IMAGES 엔티티 생성
            QcImageDTO image = new QcImageDTO();
            image.setImgName(fileName);
            image.setImgExt("mp4");
            image.setImgSize(videoFile.getSize());
            image.setImgUrl(videoUrl);
            image.setCreatedAt(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            image.setImgX(0.0);
            image.setImgY(0.0);
            image.setGhIdx(ghIdx);
            
            // DB 저장 (imgIdx 자동 생성)
            imageMapper.insertImage(image);
            
            log.info("💾 비디오 DB 저장 완료: IMG_IDX={}, URL={}", image.getImgIdx(), videoUrl);
            
            return image.getImgIdx();
            
        } catch (Exception e) {
            log.error("❌ 비디오 DB 저장 실패: {}", e.getMessage());
            return null;
        }
    }
    
    // /api/video/stream/ 경로로 비디오 스트리밍 (선택사항)
    // StaticResourceConfig에서 /videos/** 경로를 이미 처리하므로 이 메서드는 선택사항
    @GetMapping("/api/video/stream/{dateFolder}/{fileName}")
    public ResponseEntity<Resource> streamVideo(
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