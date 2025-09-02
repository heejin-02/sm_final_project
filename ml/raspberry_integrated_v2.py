#!/usr/bin/env python3
"""
라즈베리파이 통합 해충 탐지 시스템 V2
Spring Boot 연동 버전 - 기존 시스템과 완벽 호환
"""

import cv2
import numpy as np
import torch
import json
import time
import os
from datetime import datetime
from collections import defaultdict
import asyncio
import aiohttp
from pathlib import Path
import logging
import requests

# 기존 SimpleTracker와 MobileNetDetector 클래스는 동일 (생략)
from raspberry_integrated import SimpleTracker  # SimpleTracker만 import

# MobileNetDetector는 수정된 클래스 이름으로 재정의
class MobileNetDetector:
    """MobileNet 기반 해충 탐지기"""
    def __init__(self, model_path, confidence_threshold=0.6):
        self.device = torch.device('cpu')  # 라즈베리파이는 CPU 사용
        self.confidence_threshold = confidence_threshold
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 전처리 설정
        self.transform = self._get_transform()
        
        # 클래스 이름 (수정된 10종)
        self.class_names = [
            "꽃노랑총채벌레",     # 0 - 해충
            "담배가루이",         # 1 - 해충
            "복숭아혹진딧물",      # 2 - 해충
            "썩덩나무노린재",      # 3 - 해충
            "비단노린재",         # 4 - 일반곤충
            "먹노린재",           # 5 - 일반곤충
            "무잎벌",            # 6 - 일반곤충
            "배추좀나방",         # 7 - 일반곤충
            "벼룩잎벌레",         # 8 - 일반곤충
            "큰28점박이무당벌레"   # 9 - 일반곤충
        ]
    
    def _load_model(self, model_path):
        """모델 로드 (ONNX 또는 PyTorch)"""
        if model_path.endswith('.onnx'):
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        else:
            # PyTorch 모델
            from torchvision import models
            import torch.nn as nn
            
            model = models.mobilenet_v2(pretrained=False)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10)
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
    
    def _get_transform(self):
        """이미지 전처리 변환"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect(self, frame, roi=None):
        """프레임에서 해충 탐지"""
        detections = []
        h, w = frame.shape[:2]
        
        # ROI 설정 (포집기 영역)
        if roi:
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
        else:
            roi_frame = frame
            x1, y1 = 0, 0
        
        # 슬라이딩 윈도우로 탐지 (간단한 구현)
        window_size = 128
        stride = 64
        
        for y in range(0, roi_frame.shape[0] - window_size, stride):
            for x in range(0, roi_frame.shape[1] - window_size, stride):
                window = roi_frame[y:y+window_size, x:x+window_size]
                
                # 전처리 및 추론
                input_tensor = self.transform(window).unsqueeze(0)
                
                with torch.no_grad():
                    if isinstance(self.model, torch.nn.Module):
                        outputs = self.model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred_class = probs.max(1)
                    else:
                        # ONNX 추론
                        outputs = self.model.run(None, {self.model.get_inputs()[0].name: input_tensor.numpy()})
                        probs = self._softmax(outputs[0])
                        conf = probs.max()
                        pred_class = probs.argmax()
                
                # 임계값 이상만 저장
                if conf > self.confidence_threshold:
                    detections.append({
                        'bbox': [x1+x, y1+y, x1+x+window_size, y1+y+window_size],
                        'confidence': float(conf),
                        'class_id': int(pred_class),
                        'class_name': self.class_names[pred_class]
                    })
        
        return detections
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class SpringBootIntegratedMonitor:
    """Spring Boot 연동 통합 모니터링 시스템"""
    
    def __init__(self, camera_id=0, spring_url="http://192.168.219.49:8095", ml_url="http://192.168.219.49:8003"):
        self.camera_id = camera_id
        self.spring_url = spring_url
        self.ml_url = ml_url
        self.greenhouse_id = 75  # GH_IDX
        
        # 컴포넌트 초기화
        self.detector = MobileNetDetector('best_mobilenet_insect.pt')
        self.tracker = SimpleTracker()
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        # 상태 관리
        self.insect_counts = defaultdict(int)
        self.last_count = 0
        self.recording = False
        self.record_frames = []
        self.metadata = []
        
        # 알림 쿨다운 (메모리 캐시)
        self.alert_cooldown = {}
        self.cooldown_minutes = 30
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def run(self):
        """메인 실행 루프"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # 3프레임마다 탐지 (성능 최적화)
            if frame_count % 3 == 0:
                # 탐지 실행
                detections = self.detector.detect(frame)
                
                # 추적 업데이트
                tracked = self.tracker.update(detections)
                
                # 마릿수 변화 감지
                current_count = len(self.tracker.tracks)
                if current_count > self.last_count:
                    self.logger.info(f"🐛 새로운 해충 탐지! (총 {current_count}마리)")
                    
                    if not self.recording:
                        # 10초 녹화 시작
                        asyncio.create_task(self.start_recording(frame, tracked))
                
                self.last_count = current_count
                
                # 시각화 (디버깅용)
                self.visualize(frame, tracked)
            
            # ESC 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cleanup()
    
    async def start_recording(self, trigger_frame, initial_detections):
        """10초 녹화 및 Spring Boot 연동"""
        self.recording = True
        self.record_frames = []
        self.metadata = []
        
        timestamp = datetime.now()
        recording_start_time = time.time()
        
        # 10초간 녹화
        while time.time() - recording_start_time < 10:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 탐지 및 추적
            frame_detections = self.detector.detect(frame)
            tracked = self.tracker.update(frame_detections)
            
            # 프레임 저장
            self.record_frames.append(frame)
            
            # 메타데이터 수집
            for detection in tracked:
                bbox = detection['bbox']
                
                # 크롭 이미지 생성
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
                self.metadata.append({
                    'frame_id': len(self.record_frames),
                    'timestamp': (timestamp + timedelta(seconds=len(self.record_frames)/10)).isoformat(),
                    'track_id': detection['track_id'],
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': bbox,
                    'crop_image': crop  # 이미지 데이터 직접 저장
                })
            
            await asyncio.sleep(0.1)  # 10 FPS
        
        # 1단계: Spring Boot로 비디오 업로드
        video_path = await self.upload_to_spring_boot()
        
        # 2단계: ML 서버로 Open Set 처리 요청
        if video_path:
            await self.process_with_ml_server(video_path)
        
        self.recording = False
    
    async def upload_to_spring_boot(self):
        """Spring Boot VideoController로 비디오 업로드"""
        try:
            # 임시 비디오 파일 생성
            temp_video = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            h, w = self.record_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, 10.0, (w, h))
            
            for frame in self.record_frames:
                out.write(frame)
            out.release()
            
            # Spring Boot API 호출
            with open(temp_video, 'rb') as video_file:
                files = {
                    'video': ('detection.mp4', video_file, 'video/mp4')
                }
                data = {
                    'camera_id': f'cam_{self.camera_id:03d}',
                    'gh_idx': self.greenhouse_id,
                    'detection_count': len(set(d['track_id'] for d in self.metadata)),
                    'recording_start_time': time.time(),
                    'frame_count': len(self.record_frames)
                }
                
                response = requests.post(
                    f"{self.spring_url}/api/video/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    video_path = result.get('video_path')
                    img_idx = result.get('img_idx')
                    
                    self.logger.info(f"✅ Spring Boot 업로드 성공: {video_path} (IMG_IDX: {img_idx})")
                    
                    # 메타데이터에 img_idx 추가
                    for item in self.metadata:
                        item['img_idx'] = img_idx
                    
                    # 임시 파일 삭제
                    os.remove(temp_video)
                    
                    return video_path
                else:
                    self.logger.error(f"❌ Spring Boot 업로드 실패: {response.status_code}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ 업로드 중 오류: {e}")
            return None
    
    async def process_with_ml_server(self, video_path):
        """ML 서버로 Open Set 처리 및 알림 요청"""
        try:
            # 크롭 이미지와 메타데이터 전송
            async with aiohttp.ClientSession() as session:
                # 메타데이터 전송
                metadata_payload = {
                    'greenhouse_id': self.greenhouse_id,
                    'video_path': video_path,
                    'detections': []
                }
                
                # 각 탐지에 대해 Open Set 처리
                for idx, detection in enumerate(self.metadata):
                    # 크롭 이미지를 base64로 인코딩
                    crop_image = detection.pop('crop_image')
                    _, buffer = cv2.imencode('.jpg', crop_image)
                    image_base64 = buffer.tobytes().hex()
                    
                    detection['crop_base64'] = image_base64
                    metadata_payload['detections'].append(detection)
                
                # Open Set Recognition 요청
                async with session.post(
                    f"{self.ml_url}/api/process-detections",
                    json=metadata_payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        # 알림 처리
                        await self.handle_alerts(result)
                        
                        self.logger.info(f"✅ ML 처리 완료: {result.get('processed_count')}개 탐지")
                    else:
                        self.logger.error(f"❌ ML 처리 실패: {resp.status}")
                        
        except Exception as e:
            self.logger.error(f"❌ ML 서버 연동 오류: {e}")
    
    async def handle_alerts(self, ml_result):
        """알림 처리 (SignalWire 전화 등)"""
        alerts = ml_result.get('alerts', [])
        
        for alert in alerts:
            insect_type = alert['insect_type']
            confidence = alert['confidence']
            
            # 쿨다운 체크
            if self.check_cooldown(insect_type):
                # SignalWire 전화 발신 요청
                await self.make_phone_call(insect_type, confidence)
                
                # DB에 알림 기록 (Spring Boot API 통해)
                await self.save_alert_to_db(alert)
    
    def check_cooldown(self, insect_type):
        """알림 쿨다운 체크"""
        key = f"{self.greenhouse_id}_{insect_type}"
        now = datetime.now()
        
        if key in self.alert_cooldown:
            last_alert = self.alert_cooldown[key]
            if (now - last_alert).seconds < self.cooldown_minutes * 60:
                remaining = self.cooldown_minutes - (now - last_alert).seconds // 60
                self.logger.info(f"⏰ 알림 쿨다운: {insect_type} ({remaining}분 남음)")
                return False
        
        self.alert_cooldown[key] = now
        return True
    
    async def make_phone_call(self, insect_type, confidence):
        """SignalWire 전화 발신"""
        try:
            async with aiohttp.ClientSession() as session:
                # 전화번호 조회
                async with session.get(
                    f"{self.spring_url}/ml/user-phone-by-ghidx",
                    params={'gh_idx': self.greenhouse_id}
                ) as resp:
                    if resp.status == 200:
                        phone_data = await resp.json()
                        phone_number = phone_data.get('userPhone')
                        
                        # SignalWire 발신
                        call_data = {
                            'to_number': phone_number,
                            'message': f"농장에서 {insect_type}이 발견되었습니다. 신뢰도는 {confidence:.0%}입니다."
                        }
                        
                        async with session.post(
                            f"{self.ml_url}/api/make-call",
                            json=call_data
                        ) as call_resp:
                            if call_resp.status == 200:
                                self.logger.info(f"📞 전화 발신 성공: {phone_number}")
                            else:
                                self.logger.error(f"❌ 전화 발신 실패")
                                
        except Exception as e:
            self.logger.error(f"❌ 전화 발신 오류: {e}")
    
    async def save_alert_to_db(self, alert):
        """알림 DB 저장 (Spring Boot 경유)"""
        try:
            # Spring Boot의 AlertController 통해 저장
            alert_data = {
                'gh_idx': self.greenhouse_id,
                'insect_idx': alert['class_id'],
                'insect_name': alert['insect_type'],
                'confidence': alert['confidence'],
                'img_idx': alert.get('img_idx'),
                'created_at': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.spring_url}/api/alert/save",
                    json=alert_data
                ) as resp:
                    if resp.status == 200:
                        self.logger.info("💾 알림 DB 저장 완료")
                    else:
                        self.logger.error("❌ 알림 DB 저장 실패")
                        
        except Exception as e:
            self.logger.error(f"❌ DB 저장 오류: {e}")
    
    def visualize(self, frame, detections):
        """디버깅용 시각화"""
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨
            label = f"{detection['class_name']} #{detection['track_id']}"
            conf = f"{detection['confidence']:.2f}"
            cv2.putText(frame, f"{label} {conf}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 상태 표시
        status = "🔴 Recording" if self.recording else "⚪ Monitoring"
        info = f"{status} | Total: {len(self.tracker.tracks)}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Spring Boot Integrated Monitor', frame)
    
    def cleanup(self):
        """종료 처리"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("🔚 모니터링 종료")

if __name__ == "__main__":
    # 실행
    monitor = SpringBootIntegratedMonitor(
        camera_id=0,
        spring_url="http://192.168.219.49:8095",
        ml_url="http://192.168.219.49:8003"
    )
    asyncio.run(monitor.run())

# 설치 필요:
# pip install opencv-python torch torchvision aiohttp requests