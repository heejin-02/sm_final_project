package com.smhrd.web.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class StaticResourceConfig implements WebMvcConfigurer {

    @Value("${file.upload.dir}")
    private String uploadDir;

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 절대 경로로 변환
        String absolutePath = new java.io.File(uploadDir).getAbsolutePath();
        
        // Windows 경로 처리 (역슬래시를 슬래시로 변환)
        absolutePath = absolutePath.replace("\\", "/");
        
        // 비디오 파일 정적 리소스 매핑
        registry.addResourceHandler("/videos/**")
                .addResourceLocations("file:///" + absolutePath + "/")  // Windows에서는 file:/// 필요
                .setCachePeriod(3600); // 1시간 캐시
        
        System.out.println("📁 비디오 리소스 경로 매핑: /videos/** -> file:///" + absolutePath + "/");
    }
}