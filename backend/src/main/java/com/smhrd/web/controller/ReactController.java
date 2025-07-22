package com.smhrd.web.controller;

import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@RestController
@RequestMapping("/api")
public class ReactController {

	@Configuration
	public class WebConfig implements WebMvcConfigurer {
		@Override
		public void addCorsMappings(@NonNull CorsRegistry registry) {
			registry.addMapping("/**")
					.allowedOrigins("http://localhost:5173") // React 주소
					.allowedMethods("GET", "POST");
		}
	}
}
