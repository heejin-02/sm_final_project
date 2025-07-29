package com.smhrd.web.insect;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class InsectSolutionService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String FASTAPI_URL = "http://localhost:8000/summary";

    public String getSolutionSummary(String insectName) {
        try {
            // JSON body 구성
            Map<String, String> body = new HashMap<>();
            body.put("insect_name", insectName);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, String>> entity = new HttpEntity<>(body, headers);

            ResponseEntity<InsectSummaryResponseDTO> response = restTemplate.exchange(
                    FASTAPI_URL,
                    HttpMethod.POST,
                    entity,
                    InsectSummaryResponseDTO.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody().getSolution_summary();
            } else {
                return "방제 정보를 가져오는 데 실패했습니다.";
            }

        } catch (Exception e) {
            e.printStackTrace();
            return "[ERROR] FastAPI 통신 실패";
        }
    }
}
