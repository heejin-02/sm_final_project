#!/usr/bin/env python3
"""
라즈베리파이 통합 해충 탐지 시스템
MobileNet V2 + SORT 추적 + 메타데이터 생성
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

# SORT 추적기 (간소화 버전)
class SimpleTracker:
    """간단한 객체 추적기 (SORT 대체)"""
    def __init__(self, max_lost=30):
        self.max_lost = max_lost
        self.tracks = {}
        self.track_id = 0
        self.frame_count = 0
        
    def update(self, detections):
        """탐지 결과 업데이트 및 ID 할당"""
        self.frame_count += 1
        matched_tracks = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = self._get_center(bbox)
            
            # 가장 가까운 트랙 찾기
            min_dist = float('inf')
            matched_id = None
            
            for track_id, track in self.tracks.items():
                if track['lost'] > 0:
                    continue
                dist = self._calculate_distance(center, track['center'])
                if dist < min_dist and dist < 50:  # 50픽셀 임계값
                    min_dist = dist
                    matched_id = track_id
            
            if matched_id:
                # 기존 트랙 업데이트
                self.tracks[matched_id]['bbox'] = bbox
                self.tracks[matched_id]['center'] = center
                self.tracks[matched_id]['lost'] = 0
                detection['track_id'] = matched_id
            else:
                # 새 트랙 생성
                self.track_id += 1
                self.tracks[self.track_id] = {
                    'bbox': bbox,
                    'center': center,
                    'lost': 0,
                    'start_frame': self.frame_count
                }
                detection['track_id'] = self.track_id
            
            matched_tracks.append(detection)
        
        # 매칭되지 않은 트랙 처리
        for track_id in list(self.tracks.keys()):
            if track_id not in [d['track_id'] for d in matched_tracks]:
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]
        
        return matched_tracks
    
    def _get_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
        
        # 클래스 이름
        self.class_names = [
            "꽃노랑총채벌레", "담배가루이", "비단노린재", "알락수염노린재",
            "먹노린재", "무잎벌", "배추좀나방", "벼룩잎벌레",
            "복숭아혹진딧물", "큰28점박이무당벌레"
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

class InsectMonitor:
    """통합 해충 모니터링 시스템"""
    def __init__(self, camera_id=0, server_url="http://192.168.219.43:8095"):
        self.camera_id = camera_id
        self.server_url = server_url
        self.greenhouse_id = 1  # 설정 필요
        
        # 컴포넌트 초기화
        self.detector = MobileNetDetector('best_mobilenet_insect.pt')
        self.tracker = SimpleTracker()
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)  # 라즈베리파이 성능 고려
        
        # 상태 관리
        self.insect_counts = defaultdict(int)
        self.last_count = 0
        self.recording = False
        self.record_frames = []
        self.metadata = []
        
        # 로깅 설정
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
                    self.logger.info(f"새로운 해충 탐지! (총 {current_count}마리)")
                    
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
    
    async def start_recording(self, trigger_frame, detections):
        """10초 녹화 및 메타데이터 생성"""
        self.recording = True
        self.record_frames = []
        self.metadata = []
        
        timestamp = datetime.now()
        video_filename = f"detection_{self.greenhouse_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # 10초간 녹화
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 탐지 및 추적
            frame_detections = self.detector.detect(frame)
            tracked = self.tracker.update(frame_detections)
            
            # 프레임 및 메타데이터 저장
            self.record_frames.append(frame)
            
            for detection in tracked:
                # 크롭 이미지 저장
                bbox = detection['bbox']
                crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_filename = f"crop_{detection['track_id']}_{len(self.metadata)}.jpg"
                
                self.metadata.append({
                    'frame_id': len(self.record_frames),
                    'timestamp': (timestamp + datetime.timedelta(seconds=len(self.record_frames)/10)).isoformat(),
                    'track_id': detection['track_id'],
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'crop_path': crop_filename
                })
            
            await asyncio.sleep(0.1)  # 10 FPS
        
        # 서버로 전송
        await self.send_to_server(video_filename)
        
        self.recording = False
    
    async def send_to_server(self, video_filename):
        """서버로 비디오 및 메타데이터 전송"""
        try:
            # 비디오 저장
            h, w = self.record_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, 10.0, (w, h))
            
            for frame in self.record_frames:
                out.write(frame)
            out.release()
            
            # 메타데이터 저장
            metadata_filename = video_filename.replace('.mp4', '_metadata.json')
            with open(metadata_filename, 'w') as f:
                json.dump({
                    'greenhouse_id': self.greenhouse_id,
                    'video_file': video_filename,
                    'total_frames': len(self.record_frames),
                    'detections': self.metadata,
                    'insect_summary': dict(self.insect_counts)
                }, f, indent=2)
            
            # HTTP 전송
            async with aiohttp.ClientSession() as session:
                # 비디오 업로드
                with open(video_filename, 'rb') as f:
                    files = {'video': f}
                    data = {'greenhouse_id': self.greenhouse_id}
                    async with session.post(f"{self.server_url}/api/ml/upload", data=data, files=files) as resp:
                        result = await resp.json()
                        self.logger.info(f"비디오 업로드 완료: {result}")
                
                # 메타데이터 전송
                with open(metadata_filename, 'r') as f:
                    metadata = json.load(f)
                    async with session.post(f"{self.server_url}/api/ml/metadata", json=metadata) as resp:
                        result = await resp.json()
                        self.logger.info(f"메타데이터 전송 완료: {result}")
            
            # 로컬 파일 정리 (옵션)
            os.remove(video_filename)
            os.remove(metadata_filename)
            
        except Exception as e:
            self.logger.error(f"서버 전송 실패: {e}")
            # 오프라인 큐에 저장
            self.save_offline(video_filename, metadata_filename)
    
    def save_offline(self, video_file, metadata_file):
        """네트워크 오류 시 오프라인 저장"""
        offline_dir = Path("offline_queue")
        offline_dir.mkdir(exist_ok=True)
        
        # 파일 이동
        os.rename(video_file, offline_dir / video_file)
        os.rename(metadata_file, offline_dir / metadata_file)
        
        self.logger.info(f"오프라인 큐에 저장: {video_file}")
    
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
        
        # 통계 표시
        info = f"Total: {len(self.tracker.tracks)} | Recording: {self.recording}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Insect Monitor', frame)
    
    def cleanup(self):
        """종료 처리"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("모니터링 종료")

if __name__ == "__main__":
    # 실행
    monitor = InsectMonitor(camera_id=0)
    asyncio.run(monitor.run())

# 설치 필요 패키지:
# pip install opencv-python torch torchvision aiohttp onnxruntime