package com.smhrd.web.camera;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.smhrd.web.ml.MlApiService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Component
@RequiredArgsConstructor
public class CameraWebSocketHandler implements WebSocketHandler {

    private final MlApiService mlApiService;
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

    private void handlePing(WebSocketSession session) throws IOException {
        String pongResponse = "{\"type\":\"pong\",\"timestamp\":" + System.currentTimeMillis() + "}";
        session.sendMessage(new TextMessage(pongResponse));
    }

    private void forwardToMLServer(JsonNode frameData) {
        // ML 서버로 데이터 전달 (비동기 처리)
        // 현재는 로깅만 수행, 필요시 HTTP API로 ML 서버에 전송
        log.info("ML 서버로 프레임 데이터 전달 (구현 필요)");
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