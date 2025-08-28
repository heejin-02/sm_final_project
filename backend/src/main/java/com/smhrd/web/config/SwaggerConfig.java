package com.smhrd.web.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SwaggerConfig {

	@Bean
	public OpenAPI openAPI() {
		return new OpenAPI()
				.info(new Info()
						.title("벌레잡는 109")
						.description("AI 기반 해충 실시간 탐지 및 조기 알림 서비스")
						.version("1.0.0"));
	}
}