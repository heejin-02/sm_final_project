"""
Open Set Recognition 서버 모듈
2개 앙상블: 거리 기반 + 확률 기반
알림 폭탄 방지 및 영상 용량 관리 포함
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Tuple
import hashlib

class OpenSetRecognizer:
    """Open Set 해충 인식 시스템"""
    
    def __init__(self, known_classes=4, confidence_threshold=0.65):
        """
        Args:
            known_classes: 학습된 해충 종류 수 (현재 4종)
            confidence_threshold: 최종 신뢰도 임계값
        """
        self.known_classes = known_classes
        self.confidence_threshold = confidence_threshold
        
        # 알려진 해충 (학습된 4종)
        self.known_insects = {
            0: "꽃노랑총채벌레",
            1: "담배가루이",
            2: "복숭아혹진딧물",
            3: "썩덩나무노린재"
        }
        
        # 특징 추출기 로드
        self.feature_extractor = self._load_feature_extractor()
        
        # 클래스별 프로토타입 (학습 데이터의 평균 특징)
        self.class_prototypes = self._load_prototypes()
        
        # 알림 쿨다운 관리
        self.alert_cooldown = {}
        self.cooldown_minutes = 30
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_feature_extractor(self):
        """특징 추출 모델 로드"""
        from torchvision import models
        model = models.resnet50(pretrained=True)
        # 마지막 FC 레이어 제거 (특징만 추출)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model
    
    def _load_prototypes(self):
        """각 클래스의 프로토타입 특징 로드"""
        import pickle
        
        # 학습된 프로토타입 파일 확인
        prototype_file = 'pest_prototypes.pkl'
        
        if os.path.exists(prototype_file):
            # 학습된 프로토타입 로드
            with open(prototype_file, 'rb') as f:
                data = pickle.load(f)
                prototypes = data['prototypes']
                self.thresholds = data.get('thresholds', {})
                self.logger.info(f"✅ 프로토타입 로드 완료: {prototype_file}")
                return prototypes
        else:
            # 경고: 더미 값 사용
            self.logger.warning("⚠️ 프로토타입 파일 없음! 더미 값 사용 (정확도 낮음)")
            self.logger.warning(f"👉 먼저 실행: python train_prototypes.py")
            
            prototypes = {}
            for class_id in self.known_insects.keys():
                prototypes[class_id] = np.random.randn(2048)  # ResNet50 특징 크기
                prototypes[class_id] = prototypes[class_id] / np.linalg.norm(prototypes[class_id])
            return prototypes
    
    def extract_features(self, image):
        """이미지에서 특징 추출"""
        # 전처리
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, np.ndarray):
            image_tensor = transform(image).unsqueeze(0)
        else:
            image_tensor = image
        
        # 특징 추출
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = features.squeeze().numpy()
            features = features / np.linalg.norm(features)  # 정규화
        
        return features
    
    def distance_based_verification(self, features):
        """거리 기반 검증 (코사인 유사도)"""
        max_similarity = -1
        best_class = -1
        
        for class_id, prototype in self.class_prototypes.items():
            similarity = cosine_similarity([features], [prototype])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_class = class_id
        
        # 유사도를 확률로 변환 (0.5 ~ 1.0 범위)
        confidence = (max_similarity + 1) / 2
        
        return best_class, confidence, max_similarity
    
    def probability_based_verification(self, features):
        """확률 기반 검증 (소프트맥스 with temperature)"""
        temperature = 2.0  # 온도 매개변수 (불확실성 조정)
        
        # 각 클래스와의 거리 계산
        logits = []
        for class_id, prototype in self.class_prototypes.items():
            similarity = cosine_similarity([features], [prototype])[0][0]
            logits.append(similarity / temperature)
        
        # 소프트맥스
        logits = np.array(logits)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        best_class = np.argmax(probs)
        confidence = probs[best_class]
        
        return best_class, confidence, probs
    
    def ensemble_prediction(self, image):
        """앙상블 예측 (2개 방법 조합)"""
        # 특징 추출
        features = self.extract_features(image)
        
        # 1. 거리 기반
        dist_class, dist_conf, similarity = self.distance_based_verification(features)
        
        # 2. 확률 기반
        prob_class, prob_conf, probs = self.probability_based_verification(features)
        
        # 앙상블 결정
        if dist_class == prob_class:
            # 두 방법이 일치
            final_class = dist_class
            final_confidence = (dist_conf + prob_conf) / 2
        else:
            # 불일치 시 더 높은 신뢰도 선택
            if dist_conf > prob_conf:
                final_class = dist_class
                final_confidence = dist_conf * 0.7  # 페널티 적용
            else:
                final_class = prob_class
                final_confidence = prob_conf * 0.7
        
        # Unknown 처리
        if final_confidence < self.confidence_threshold:
            return {
                'class_id': -1,
                'class_name': '미확인 해충',
                'confidence': final_confidence,
                'is_known': False,
                'details': {
                    'distance_based': {'class': dist_class, 'confidence': dist_conf},
                    'probability_based': {'class': prob_class, 'confidence': prob_conf}
                }
            }
        
        return {
            'class_id': final_class,
            'class_name': self.known_insects.get(final_class, '미확인'),
            'confidence': final_confidence,
            'is_known': True,
            'details': {
                'distance_based': {'class': dist_class, 'confidence': dist_conf},
                'probability_based': {'class': prob_class, 'confidence': prob_conf}
            }
        }
    
    def check_alert_cooldown(self, greenhouse_id, insect_type):
        """알림 쿨다운 체크"""
        key = f"{greenhouse_id}_{insect_type}"
        now = datetime.now()
        
        if key in self.alert_cooldown:
            last_alert = self.alert_cooldown[key]
            if now - last_alert < timedelta(minutes=self.cooldown_minutes):
                remaining = self.cooldown_minutes - (now - last_alert).seconds // 60
                self.logger.info(f"알림 쿨다운 중: {insect_type} ({remaining}분 남음)")
                return False
        
        self.alert_cooldown[key] = now
        return True
    
    def process_video_metadata(self, metadata_path):
        """비디오 메타데이터 처리 및 Open Set 분류"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        greenhouse_id = metadata['greenhouse_id']
        detections = metadata['detections']
        
        # 각 탐지에 대해 Open Set 분류
        processed_detections = []
        alert_queue = []
        
        for detection in detections:
            crop_path = detection['crop_path']
            
            if os.path.exists(crop_path):
                # 이미지 품질 체크
                if not self.check_image_quality(crop_path):
                    continue
                
                # Open Set 분류
                crop_image = cv2.imread(crop_path)
                result = self.ensemble_prediction(crop_image)
                
                # 메타데이터 업데이트
                detection['open_set_result'] = result
                
                # 해충이 확인된 경우
                if result['is_known'] and result['confidence'] > 0.7:
                    insect_type = result['class_name']
                    
                    # 알림 쿨다운 체크
                    if self.check_alert_cooldown(greenhouse_id, insect_type):
                        alert_queue.append({
                            'greenhouse_id': greenhouse_id,
                            'insect_type': insect_type,
                            'confidence': result['confidence'],
                            'detection': detection
                        })
                    
                    processed_detections.append(detection)
                else:
                    # 해충이 아닌 경우 이미지 삭제
                    os.remove(crop_path)
                    self.logger.info(f"비해충 이미지 삭제: {crop_path}")
            
        return processed_detections, alert_queue
    
    def check_image_quality(self, image_path, threshold=100):
        """Laplacian variance를 이용한 이미지 품질 체크"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance > threshold
    
    def create_alert_video(self, original_video, detections, insect_type):
        """특정 해충만 바운딩박스로 표시한 영상 생성"""
        cap = cv2.VideoCapture(original_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 비디오 설정
        output_path = f"alert_{insect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 해당 해충의 탐지만 필터링
        target_detections = [d for d in detections 
                            if d.get('open_set_result', {}).get('class_name') == insect_type]
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 프레임의 탐지 찾기
            frame_detections = [d for d in target_detections if d['frame_id'] == frame_id]
            
            # 바운딩 박스 그리기
            for detection in frame_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 라벨
                label = f"{insect_type} #{detection['track_id']}"
                conf = detection['open_set_result']['confidence']
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
            frame_id += 1
        
        cap.release()
        out.release()
        
        return output_path

class VideoStorageManager:
    """영상 저장 용량 관리"""
    
    def __init__(self, storage_path="./videos", max_days=7, max_size_gb=50):
        self.storage_path = Path(storage_path)
        self.max_days = max_days
        self.max_size_gb = max_size_gb
        self.logger = logging.getLogger(__name__)
    
    def cleanup_old_videos(self):
        """오래된 비디오 삭제"""
        now = datetime.now()
        total_deleted = 0
        
        for video_file in self.storage_path.glob("**/*.mp4"):
            # 파일 생성 시간 확인
            file_time = datetime.fromtimestamp(video_file.stat().st_mtime)
            
            if now - file_time > timedelta(days=self.max_days):
                file_size = video_file.stat().st_size
                video_file.unlink()
                total_deleted += file_size
                self.logger.info(f"오래된 비디오 삭제: {video_file.name}")
        
        if total_deleted > 0:
            self.logger.info(f"총 {total_deleted / 1024 / 1024:.2f} MB 삭제")
    
    def check_storage_usage(self):
        """저장 공간 사용량 체크"""
        total_size = sum(f.stat().st_size for f in self.storage_path.glob("**/*") if f.is_file())
        total_gb = total_size / 1024 / 1024 / 1024
        
        if total_gb > self.max_size_gb:
            self.logger.warning(f"저장 공간 초과: {total_gb:.2f} GB / {self.max_size_gb} GB")
            # 가장 오래된 파일부터 삭제
            self.cleanup_by_size()
        
        return total_gb
    
    def cleanup_by_size(self):
        """용량 기준으로 정리"""
        files = sorted(self.storage_path.glob("**/*.mp4"), 
                      key=lambda f: f.stat().st_mtime)
        
        total_size = sum(f.stat().st_size for f in files)
        target_size = self.max_size_gb * 1024 * 1024 * 1024
        
        for file in files:
            if total_size <= target_size:
                break
            
            file_size = file.stat().st_size
            file.unlink()
            total_size -= file_size
            self.logger.info(f"용량 관리 삭제: {file.name}")
    
    def compress_video(self, input_path, quality=23):
        """H.264 압축"""
        output_path = input_path.replace(".mp4", "_compressed.mp4")
        
        import subprocess
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-crf', str(quality),
            '-preset', 'fast', '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 원본 삭제 및 교체
            os.remove(input_path)
            os.rename(output_path, input_path)
            
            self.logger.info(f"비디오 압축 완료: {input_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"압축 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # Open Set 인식기 초기화
    recognizer = OpenSetRecognizer(known_classes=4)
    
    # 비디오 저장 관리자
    storage_manager = VideoStorageManager()
    
    # 메타데이터 처리
    metadata_path = "detection_metadata.json"
    processed, alerts = recognizer.process_video_metadata(metadata_path)
    
    # 알림 영상 생성
    for alert in alerts:
        video_path = recognizer.create_alert_video(
            "original_video.mp4",
            processed,
            alert['insect_type']
        )
        print(f"알림 영상 생성: {video_path}")
    
    # 저장 공간 관리
    storage_manager.cleanup_old_videos()
    usage = storage_manager.check_storage_usage()
    print(f"현재 저장 공간 사용: {usage:.2f} GB")