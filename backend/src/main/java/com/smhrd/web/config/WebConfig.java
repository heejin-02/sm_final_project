//React(포트 5173)에서 내 Spring 서버(포트 8080)에 요청할 때 “보안상 막지 말아줘!”라고 알려주는 역할
package com.smhrd.web.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
	@Override
	public void addCorsMappings(CorsRegistry registry) {
		registry.addMapping("/api/**") // /api 로 시작하는 모든 요청에 대해
				.allowedOrigins("http://localhost:5173") // React 앱 주소 허용
				.allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
				.allowCredentials(true); // 쿠키(세션) 주고받기 허용
	}
}
