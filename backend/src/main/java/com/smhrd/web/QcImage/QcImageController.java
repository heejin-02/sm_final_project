package com.smhrd.web.QcImage;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/qc-videos")
public class QcImageController {

    @Value("${file.upload.dir}")
    private String uploadDir;

    private final QcImageMapper imageMapper;

    public QcImageController(QcImageMapper imageMapper) {
        this.imageMapper = imageMapper;
    }
    
   

    @PostMapping
    public ResponseEntity<Map<String, Object>> uploadVideo(@RequestParam("video") MultipartFile file, @RequestParam("classId") int classId) {
        Map<String, Object> response = new HashMap<>();

        try {
            // 1. 날짜별 폴더 생성
        	String dateFolder = java.time.LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));

            File folder = new File(uploadDir, dateFolder);
            if (!folder.exists()) {
                folder.mkdirs();
            }
            
            // 2. 파일명 생성
            String date = dateFolder;
            String time = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmmss"));
            String fileName = classId + "_" + date + "_" + time + ".mp4";

            // 2. 파일 저장
 
            String filePath = new File(folder, fileName).getAbsolutePath();
            System.out.println("저장될 파일 경로: " + filePath);
            file.transferTo(new File(filePath));
            System.out.println("파일 저장 완료!");


            // 3. DB 저장용 정보 구성
            String videoUrl = "/" + dateFolder + "/" + fileName;

            QcImageDTO image = new QcImageDTO();
            image.setCctvIdx(1L); // 임시 카메라 번호
            image.setImgName(fileName);
            image.setImgExt("mp4");
            image.setImgSize(file.getSize());
            image.setImgUrl(videoUrl);
            image.setCreatedAt(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            image.setImgX(0.0);  // 필요 시 설정
            image.setImgY(0.0);  // 필요 시 설정

            // 4. DB 저장
            imageMapper.insertImage(image);  // imgIdx 자동 주입됨 (keyProperty)

            // 5. 응답 반환
            response.put("imgIdx", image.getImgIdx());
            response.put("videoUrl", videoUrl);
            return ResponseEntity.ok(response);

        } catch (IOException e) {
            e.printStackTrace();
            response.put("error", "업로드 실패");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}
