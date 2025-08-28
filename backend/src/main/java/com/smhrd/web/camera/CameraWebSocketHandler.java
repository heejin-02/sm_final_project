package com.smhrd.web.camera;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.smhrd.web.ml.MlApiService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Component
@RequiredArgsConstructor
public class CameraWebSocketHandler implements WebSocketHandler {

    private final MlApiService mlApiService;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    // 카메라 연결 관리
    private final ConcurrentHashMap<String, CameraSession> cameraSessions = new ConcurrentHashMap<>();
    
    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        log.info("카메라 웹소켓 연결 설정됨: {}", session.getId());
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) throws Exception {
        if (message instanceof TextMessage) {
            handleTextMessage(session, (TextMessage) message);
        }
    }

    private void handleTextMessage(WebSocketSession session, TextMessage message) throws IOException {
        try {
            JsonNode jsonNode = objectMapper.readTree(message.getPayload());
            String messageType = jsonNode.get("type").asText();
            
            switch (messageType) {
                case "camera_init":
                    handleCameraInit(session, jsonNode);
                    break;
                case "frame_data":
                    handleFrameData(session, jsonNode);
                    break;
                case "recording_event":
                    handleRecordingEvent(session, jsonNode);
                    break;
                case "video_buffer":
                    handleVideoBuffer(session, jsonNode);
                    break;
                case "ping":
                    handlePing(session);
                    break;
                default:
                    log.warn("알 수 없는 메시지 타입: {}", messageType);
            }
        } catch (Exception e) {
            log.error("메시지 처리 중 오류 발생", e);
        }
    }

    private void handleCameraInit(WebSocketSession session, JsonNode message) throws IOException {
        String cameraId = message.get("camera_id").asText();
        Long ghIdx = message.get("gh_idx").asLong();
        
        CameraSession cameraSession = new CameraSession();
        cameraSession.setSession(session);
        cameraSession.setCameraId(cameraId);
        cameraSession.setGhIdx(ghIdx);
        cameraSession.setConnectedAt(System.currentTimeMillis());
        
        cameraSessions.put(session.getId(), cameraSession);
        
        // 연결 확인 응답
        String response = objectMapper.writeValueAsString(new ConnectionResponse("connection_confirmed", cameraId));
        session.sendMessage(new TextMessage(response));
        
        log.info("카메라 초기화 완료: {} (GH_IDX: {})", cameraId, ghIdx);
    }

    private void handleFrameData(WebSocketSession session, JsonNode message) throws IOException {
        CameraSession cameraSession = cameraSessions.get(session.getId());
        if (cameraSession == null) return;
        
        String frameType = message.get("frame_type").asText();
        boolean motionDetected = message.get("motion_detected").asBoolean();
        
        // HQ 프레임이고 움직임이 감지되었을 때만 ML 서버로 전달
        if ("hq".equals(frameType) && motionDetected) {
            forwardToMLServer(message);
        }
        
        cameraSession.setFrameCount(cameraSession.getFrameCount() + 1);
        cameraSession.setLastFrameTime(System.currentTimeMillis());
        
        // 통계 로깅
        if (cameraSession.getFrameCount() % 100 == 0) {
            long uptime = System.currentTimeMillis() - cameraSession.getConnectedAt();
            double fps = cameraSession.getFrameCount() * 1000.0 / uptime;
            log.info("카메라 {} 통계: FPS {:.1f}, 총 프레임 {}", 
                    cameraSession.getCameraId(), fps, cameraSession.getFrameCount());
        }
    }

    private void handleRecordingEvent(WebSocketSession session, JsonNode message) {
        CameraSession cameraSession = cameraSessions.get(session.getId());
        if (cameraSession == null) return;
        
        String eventType = message.get("event_type").asText();
        log.info("녹화 이벤트: {} -> {}", cameraSession.getCameraId(), eventType);
    }

    private void handleVideoBuffer(WebSocketSession session, JsonNode message) {
        CameraSession cameraSession = cameraSessions.get(session.getId());
        if (cameraSession == null) return;
        
        String cameraId = message.get("camera_id").asText();
        Long ghIdx = message.get("gh_idx").asLong();
        int frameCount = message.get("frame_count").asInt();
        
        log.info("🎬 비디오 버퍼 수신: camera={}, frames={}, gh_idx={}", cameraId, frameCount, ghIdx);
        
        // ML 서버로 비동기 전송
        forwardVideoBufferToMLServer(message);
    }

    private void handlePing(WebSocketSession session) throws IOException {
        String pongResponse = "{\"type\":\"pong\",\"timestamp\":" + System.currentTimeMillis() + "}";
        session.sendMessage(new TextMessage(pongResponse));
    }

    private void forwardToMLServer(JsonNode frameData) {
        // ML 서버로 데이터 전달 (비동기 처리)
        try {
            String mlApiUrl = "http://localhost:8003/api/detect";
            
            // ML 서버 전송용 데이터 구성
            Map<String, Object> detectRequest = new HashMap<>();
            detectRequest.put("frame_data", frameData.get("frame_data").asText());
            detectRequest.put("camera_id", frameData.get("camera_id").asText());
            detectRequest.put("gh_idx", frameData.get("gh_idx").asLong());
            detectRequest.put("timestamp", frameData.get("timestamp").asDouble());
            
            HttpHeaders headers = createHeaders();
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(detectRequest, headers);
            
            // 비동기 호출 (별도 스레드에서 실행)
            CompletableFuture.runAsync(() -> {
                try {
                    ResponseEntity<Map> response = restTemplate.postForEntity(mlApiUrl, request, Map.class);
                    if (response.getStatusCode().is2xxSuccessful()) {
                        log.info("✅ ML 서버 탐지 요청 성공: {}", response.getBody());
                        
                        // 탐지 결과가 있으면 처리
                        processDetectionResult(response.getBody(), frameData);
                    } else {
                        log.warn("⚠️ ML 서버 응답 오류: {}", response.getStatusCode());
                    }
                } catch (Exception e) {
                    log.error("❌ ML 서버 전송 실패: {}", e.getMessage());
                }
            });
            
        } catch (Exception e) {
            log.error("❌ ML 서버 요청 구성 실패: {}", e.getMessage());
        }
    }
    
    private void forwardVideoBufferToMLServer(JsonNode videoBuffer) {
        // ML 서버로 비디오 버퍼 전송 (비동기 처리)
        CompletableFuture.runAsync(() -> {
            try {
                String mlApiUrl = "http://localhost:8003/api/process-video-buffer";
                
                HttpHeaders headers = createHeaders();
                
                // JsonNode를 Map으로 변환하여 전송
                Map<String, Object> requestBody = objectMapper.convertValue(videoBuffer, Map.class);
                HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
                
                log.info("📤 ML 서버로 비디오 버퍼 전송 중...");
                
                ResponseEntity<Map> response = restTemplate.postForEntity(mlApiUrl, request, Map.class);
                
                if (response.getStatusCode().is2xxSuccessful()) {
                    Map<String, Object> result = response.getBody();
                    log.info("✅ ML 서버 처리 완료: {}", result);
                    
                    // 탐지 결과 처리
                    processVideoDetectionResult(result);
                } else {
                    log.warn("⚠️ ML 서버 응답 오류: {}", response.getStatusCode());
                }
                
            } catch (Exception e) {
                log.error("❌ ML 서버 비디오 버퍼 전송 실패: {}", e.getMessage());
            }
        });
    }

    private HttpHeaders createHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return headers;
    }
    
    private void processDetectionResult(Map<String, Object> detectionResult, JsonNode frameData) {
        try {
            // 탐지 결과가 있으면 추가 처리
            if (detectionResult != null && detectionResult.containsKey("detections")) {
                log.info("🐛 해충 탐지 결과: {}", detectionResult);
                
                // 필요시 알림 발송, DB 저장 등의 추가 로직 구현
                // mlApiService를 통해 처리하거나 직접 구현
            }
        } catch (Exception e) {
            log.error("탐지 결과 처리 중 오류: {}", e.getMessage());
        }
    }
    
    private void processVideoDetectionResult(Map<String, Object> result) {
        try {
            if (result != null) {
                log.info("🎬 비디오 처리 결과: {}", result);
                
                // 탐지된 곤충들이 있는지 확인
                if (result.containsKey("detections") && result.get("detections") != null) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> detections = (List<Map<String, Object>>) result.get("detections");
                    
                    log.info("🐛 총 {}마리 해충 탐지됨", detections.size());
                    
                    // 각 탐지 결과 처리 (DB 저장, 알림 등)
                    for (Map<String, Object> detection : detections) {
                        String insectName = (String) detection.get("class_name");
                        Double confidence = (Double) detection.get("confidence");
                        
                        log.info("- {}: {:.2f}%", insectName, confidence * 100);
                    }
                    
                    // 추가 처리 로직 (알림 발송, 통계 업데이트 등)
                }
            }
        } catch (Exception e) {
            log.error("비디오 탐지 결과 처리 중 오류: {}", e.getMessage());
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        log.error("웹소켓 전송 오류: {}", session.getId(), exception);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus closeStatus) throws Exception {
        CameraSession cameraSession = cameraSessions.remove(session.getId());
        if (cameraSession != null) {
            log.info("카메라 연결 해제됨: {} ({})", cameraSession.getCameraId(), closeStatus);
        }
    }

    @Override
    public boolean supportsPartialMessages() {
        return false;
    }

    // 응답 클래스들
    private static class ConnectionResponse {
        private String type;
        private String camera_id;
        
        public ConnectionResponse(String type, String cameraId) {
            this.type = type;
            this.camera_id = cameraId;
        }
        
        // getters and setters
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getCamera_id() { return camera_id; }
        public void setCamera_id(String camera_id) { this.camera_id = camera_id; }
    }

    // 카메라 세션 정보
    private static class CameraSession {
        private WebSocketSession session;
        private String cameraId;
        private Long ghIdx;
        private long connectedAt;
        private long frameCount;
        private long lastFrameTime;
        
        // getters and setters
        public WebSocketSession getSession() { return session; }
        public void setSession(WebSocketSession session) { this.session = session; }
        public String getCameraId() { return cameraId; }
        public void setCameraId(String cameraId) { this.cameraId = cameraId; }
        public Long getGhIdx() { return ghIdx; }
        public void setGhIdx(Long ghIdx) { this.ghIdx = ghIdx; }
        public long getConnectedAt() { return connectedAt; }
        public void setConnectedAt(long connectedAt) { this.connectedAt = connectedAt; }
        public long getFrameCount() { return frameCount; }
        public void setFrameCount(long frameCount) { this.frameCount = frameCount; }
        public long getLastFrameTime() { return lastFrameTime; }
        public void setLastFrameTime(long lastFrameTime) { this.lastFrameTime = lastFrameTime; }
    }
}