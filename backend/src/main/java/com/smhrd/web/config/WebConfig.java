package com.smhrd.web.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.lang.NonNull;
import org.springframework.web.servlet.config.annotation.ContentNegotiationConfigurer;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    // mp4 파일의 MIME 타입 설정
    @Override
    public void configureContentNegotiation(ContentNegotiationConfigurer configurer) {
        configurer.mediaType("mp4", MediaType.valueOf("video/mp4"));
    }

    // 로컬 비디오 파일 경로 매핑
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/videos/**")  // 클라이언트가 접근할 경로
                .addResourceLocations("file:///C:/Users/smhrd1/Desktop/videos/")  // 실제 파일 위치
                .setCachePeriod(3600); // 캐시 설정
    }

    // CORS 설정
    @Override
    public void addCorsMappings(@NonNull CorsRegistry registry) {
        registry.addMapping("/**")  // 전체 경로 허용
                .allowedOrigins("http://localhost:5173",
                		"smfinalproject-production.up.railway.app") // 프론트엔드 주소 (필요시 여러개 가능)
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowCredentials(true); // 인증 정보 허용
    }
}
