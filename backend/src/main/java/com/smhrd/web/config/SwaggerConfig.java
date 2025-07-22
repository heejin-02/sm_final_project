package com.smhrd.web.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SwaggerConfig {

	@Bean
	public OpenAPI apiInfo() {
		return new OpenAPI()
				.info(new Info()
						.title("벌레잡는109(백구) API 문서")
						.description("Spring Boot 기반 해충 탐지 API")
						.version("v1.0.0"));
	}
}
