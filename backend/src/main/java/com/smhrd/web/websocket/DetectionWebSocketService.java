package com.smhrd.web.websocket;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class DetectionWebSocketService {

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    /**
     * 해충 탐지 시 해당 사용자에게 알림 전송
     */
    public void sendDetectionAlert(String userPhone, Long alertId) {
        Map<String, Object> message = new HashMap<>();
        message.put("type", "DETECTION_ALERT");
        message.put("alertId", alertId);
        message.put("timestamp", System.currentTimeMillis());
        
        // 특정 사용자(전화번호)에게 전송
        messagingTemplate.convertAndSend("/topic/user/" + userPhone, message);
    }
}