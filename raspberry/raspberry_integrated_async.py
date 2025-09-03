#!/usr/bin/env python3
"""
라즈베리파이 통합 해충 탐지 시스템 - 완전 비동기 버전
성능 최적화를 위한 비동기 처리
"""

import cv2
import numpy as np
import torch
import json
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

# 기존 import
from raspberry_integrated import SimpleTracker

class AsyncMobileNetDetector:
    """비동기 MobileNet 탐지기"""
    def __init__(self, model_path, confidence_threshold=0.6):
        self.device = torch.device('cpu')
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_path)
        self.model.eval()
        self.transform = self._get_transform()
        
        # 클래스 이름
        self.class_names = [
            "꽃노랑총채벌레", "담배가루이", "비단노린재", "알락수염노린재",
            "먹노린재", "무잎벌", "배추좀나방", "벼룩잎벌레",
            "복숭아혹진딧물", "큰28점박이무당벌레"
        ]
        
        # CPU 집약적 작업용 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _load_model(self, model_path):
        """모델 로드"""
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
        """이미지 전처리"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _detect_sync(self, frame, roi=None):
        """동기 탐지 (스레드에서 실행)"""
        detections = []
        h, w = frame.shape[:2]
        
        if roi:
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
        else:
            roi_frame = frame
            x1, y1 = 0, 0
        
        # 슬라이딩 윈도우
        window_size = 128
        stride = 64
        
        for y in range(0, roi_frame.shape[0] - window_size, stride):
            for x in range(0, roi_frame.shape[1] - window_size, stride):
                window = roi_frame[y:y+window_size, x:x+window_size]
                input_tensor = self.transform(window).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_class = probs.max(1)
                
                if conf > self.confidence_threshold:
                    detections.append({
                        'bbox': [x1+x, y1+y, x1+x+window_size, y1+y+window_size],
                        'confidence': float(conf),
                        'class_id': int(pred_class),
                        'class_name': self.class_names[pred_class]
                    })
        
        return detections
    
    async def detect_async(self, frame, roi=None):
        """비동기 탐지"""
        loop = asyncio.get_event_loop()
        # CPU 집약적 작업을 스레드풀에서 실행
        detections = await loop.run_in_executor(
            self.executor,
            self._detect_sync,
            frame,
            roi
        )
        return detections

class AsyncVideoWriter:
    """비동기 비디오 저장"""
    def __init__(self):
        self.write_queue = Queue()
        self.writer_thread = None
        self.writer = None
        
    async def start(self, filename, fps, size):
        """비동기 쓰기 시작"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._start_sync, filename, fps, size)
    
    def _start_sync(self, filename, fps, size):
        """동기 쓰기 시작"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, size)
        
        # 쓰기 스레드 시작
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.start()
    
    def _writer_worker(self):
        """백그라운드 쓰기 워커"""
        while True:
            frame = self.write_queue.get()
            if frame is None:  # 종료 신호
                break
            self.writer.write(frame)
    
    async def write_frame(self, frame):
        """프레임 추가 (논블로킹)"""
        self.write_queue.put(frame)
    
    async def finish(self):
        """쓰기 완료"""
        self.write_queue.put(None)  # 종료 신호
        if self.writer_thread:
            self.writer_thread.join()
        if self.writer:
            self.writer.release()

class AsyncSpringBootMonitor:
    """완전 비동기 Spring Boot 연동 모니터"""
    
    def __init__(self, camera_id=0, spring_url="http://192.168.219.49:8095", ml_url="http://192.168.219.49:8003"):
        self.camera_id = camera_id
        self.spring_url = spring_url
        self.ml_url = ml_url
        self.greenhouse_id = 75
        
        # 비동기 컴포넌트
        self.detector = AsyncMobileNetDetector('best_mobilenet_insect.pt')
        self.tracker = SimpleTracker()
        
        # 카메라
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        # 상태
        self.insect_counts = defaultdict(int)
        self.last_count = 0
        self.recording = False
        self.record_frames = []
        self.metadata = []
        
        # 알림 쿨다운
        self.alert_cooldown = {}
        self.cooldown_minutes = 30
        
        # 프레임 읽기용 스레드풀
        self.frame_executor = ThreadPoolExecutor(max_workers=1)
        
        # HTTP 세션 (재사용)
        self.session = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def read_frame_async(self):
        """비동기 프레임 읽기"""
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(
            self.frame_executor,
            self.cap.read
        )
        return ret, frame
    
    async def run(self):
        """메인 실행 루프"""
        # HTTP 세션 생성
        self.session = aiohttp.ClientSession()
        
        frame_count = 0
        
        try:
            while True:
                # 비동기 프레임 읽기
                ret, frame = await self.read_frame_async()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # 3프레임마다 탐지
                if frame_count % 3 == 0:
                    # 비동기 탐지
                    detections = await self.detector.detect_async(frame)
                    
                    # 추적 업데이트 (빠른 연산)
                    tracked = self.tracker.update(detections)
                    
                    # 마릿수 변화 감지
                    current_count = len(self.tracker.tracks)
                    if current_count > self.last_count:
                        self.logger.info(f"🐛 새로운 해충 탐지! (총 {current_count}마리)")
                        
                        if not self.recording:
                            # 비동기 녹화 시작 (논블로킹)
                            asyncio.create_task(self.start_recording(frame, tracked))
                    
                    self.last_count = current_count
                    
                    # 시각화 (선택사항)
                    # await self.visualize_async(frame, tracked)
                
                # 키 입력 체크 (논블로킹)
                await asyncio.sleep(0.001)
                
        finally:
            await self.cleanup()
    
    async def start_recording(self, trigger_frame, initial_detections):
        """비동기 10초 녹화"""
        self.recording = True
        self.record_frames = []
        self.metadata = []
        
        timestamp = datetime.now()
        recording_start_time = time.time()
        
        # 비동기 비디오 쓰기 준비
        video_writer = AsyncVideoWriter()
        temp_video = f"temp_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = trigger_frame.shape[:2]
        await video_writer.start(temp_video, 10.0, (w, h))
        
        # 10초간 녹화
        while time.time() - recording_start_time < 10:
            ret, frame = await self.read_frame_async()
            if not ret:
                continue
            
            # 비동기 프레임 쓰기
            await video_writer.write_frame(frame)
            self.record_frames.append(frame)
            
            # 비동기 탐지 (백그라운드)
            if len(self.record_frames) % 5 == 0:  # 5프레임마다
                detections = await self.detector.detect_async(frame)
                tracked = self.tracker.update(detections)
                
                for detection in tracked:
                    bbox = detection['bbox']
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
                        'crop_image': crop
                    })
            
            await asyncio.sleep(0.05)  # 다른 태스크에 양보
        
        # 비디오 쓰기 완료
        await video_writer.finish()
        
        # 병렬 처리: 업로드와 ML 처리
        upload_task = asyncio.create_task(self.upload_to_spring_boot_async(temp_video))
        
        # 업로드 완료 대기
        video_path = await upload_task
        
        if video_path:
            # ML 처리 (비동기)
            await self.process_with_ml_server_async(video_path)
        
        # 임시 파일 삭제
        try:
            os.remove(temp_video)
        except:
            pass
        
        self.recording = False
    
    async def upload_to_spring_boot_async(self, video_file):
        """완전 비동기 Spring Boot 업로드"""
        try:
            # aiofiles로 비동기 파일 읽기
            async with aiofiles.open(video_file, 'rb') as f:
                video_data = await f.read()
            
            # FormData 생성
            data = aiohttp.FormData()
            data.add_field('video',
                          video_data,
                          filename='detection.mp4',
                          content_type='video/mp4')
            data.add_field('camera_id', f'cam_{self.camera_id:03d}')
            data.add_field('gh_idx', str(self.greenhouse_id))
            data.add_field('detection_count', str(len(set(d['track_id'] for d in self.metadata))))
            data.add_field('recording_start_time', str(time.time()))
            data.add_field('frame_count', str(len(self.record_frames)))
            
            # 비동기 HTTP POST
            async with self.session.post(
                f"{self.spring_url}/api/video/upload",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    video_path = result.get('video_path')
                    img_idx = result.get('img_idx')
                    
                    self.logger.info(f"✅ Spring Boot 업로드 성공: {video_path}")
                    
                    for item in self.metadata:
                        item['img_idx'] = img_idx
                    
                    return video_path
                else:
                    self.logger.error(f"❌ 업로드 실패: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ 업로드 오류: {e}")
            return None
    
    async def process_with_ml_server_async(self, video_path):
        """비동기 ML 서버 처리"""
        try:
            metadata_payload = {
                'greenhouse_id': self.greenhouse_id,
                'video_path': video_path,
                'detections': []
            }
            
            # 크롭 이미지 처리
            for detection in self.metadata[:100]:  # 최대 100개
                crop_image = detection.pop('crop_image')
                _, buffer = cv2.imencode('.jpg', crop_image)
                image_base64 = buffer.tobytes().hex()
                detection['crop_base64'] = image_base64
                metadata_payload['detections'].append(detection)
            
            # 비동기 POST
            async with self.session.post(
                f"{self.ml_url}/api/process-detections",
                json=metadata_payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # 알림 처리 (비동기)
                    await self.handle_alerts_async(result)
                    
                    self.logger.info(f"✅ ML 처리 완료")
                else:
                    self.logger.error(f"❌ ML 처리 실패: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"❌ ML 오류: {e}")
    
    async def handle_alerts_async(self, ml_result):
        """비동기 알림 처리"""
        alerts = ml_result.get('alerts', [])
        
        # 병렬 알림 처리
        alert_tasks = []
        for alert in alerts:
            if self.check_cooldown(alert['insect_type']):
                alert_tasks.append(
                    asyncio.create_task(self.send_alert_async(alert))
                )
        
        # 모든 알림 완료 대기
        if alert_tasks:
            await asyncio.gather(*alert_tasks)
    
    async def send_alert_async(self, alert):
        """비동기 알림 발송"""
        try:
            # 전화번호 조회
            async with self.session.get(
                f"{self.spring_url}/ml/user-phone-by-ghidx",
                params={'gh_idx': self.greenhouse_id}
            ) as resp:
                if resp.status == 200:
                    phone_data = await resp.json()
                    
                    # SignalWire 발신
                    call_data = {
                        'to_number': phone_data.get('userPhone'),
                        'message': f"{alert['insect_type']} 발견. 신뢰도 {alert['confidence']:.0%}"
                    }
                    
                    async with self.session.post(
                        f"{self.ml_url}/api/make-call",
                        json=call_data
                    ) as call_resp:
                        if call_resp.status == 200:
                            self.logger.info(f"📞 알림 발송 완료")
                            
        except Exception as e:
            self.logger.error(f"❌ 알림 오류: {e}")
    
    def check_cooldown(self, insect_type):
        """쿨다운 체크"""
        key = f"{self.greenhouse_id}_{insect_type}"
        now = datetime.now()
        
        if key in self.alert_cooldown:
            last_alert = self.alert_cooldown[key]
            if (now - last_alert).seconds < self.cooldown_minutes * 60:
                return False
        
        self.alert_cooldown[key] = now
        return True
    
    async def cleanup(self):
        """정리"""
        if self.session:
            await self.session.close()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 스레드풀 종료
        self.detector.executor.shutdown(wait=False)
        self.frame_executor.shutdown(wait=False)
        
        self.logger.info("🔚 종료")

if __name__ == "__main__":
    monitor = AsyncSpringBootMonitor()
    asyncio.run(monitor.run())

# 추가 필요 패키지:
# pip install aiofiles aiohttp