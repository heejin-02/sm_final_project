// 배포 상태에서, 사용자가 /admin이나 /select-farm 주소로 직접 들어와도 React 앱의 index.html을 보여주도록
package com.smhrd.web.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class SpaWebConfig implements WebMvcConfigurer {
	@Override
	public void addViewControllers(ViewControllerRegistry registry) {
		// /admin, /select-farm 같은 경로를 모두 index.html로 연결
		// 단일 세그먼트일 때
		registry.addViewController("/{spring:\\w+}")
				.setViewName("forward:/index.html");
		// 그 뒤로 이어지는 모든 경로 잡기(뒷단에 **)
		registry.addViewController("/{spring:\\w+}/**")
				.setViewName("forward:/index.html");
	}
}
