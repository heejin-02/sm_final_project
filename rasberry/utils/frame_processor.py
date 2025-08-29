"""
프레임 처리 및 압축 모듈
"""

import cv2
import numpy as np
import base64
import time
from typing import Tuple, Optional

class FrameProcessor:
    def __init__(self, max_frame_size: int = 50 * 1024, auto_quality_adjust: bool = True):
        self.max_frame_size = max_frame_size
        self.auto_quality_adjust = auto_quality_adjust
        self.current_quality = 60
        self.quality_history = []
        self.bandwidth_stats = {
            'last_frame_size': 0,
            'avg_frame_size': 0,
            'total_frames': 0
        }
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """프레임 리사이즈"""
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    def compress_frame(self, frame: np.ndarray, quality: int = 60) -> Tuple[bytes, int]:
        """
        프레임을 JPEG로 압축
        
        Returns:
            (compressed_data, actual_size)
        """
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        compressed_data = buffer.tobytes()
        
        return compressed_data, len(compressed_data)
    
    def encode_frame_base64(self, frame: np.ndarray, quality: int = 60, convert_to_rgb: bool = False) -> Tuple[str, int]:
        """
        프레임을 Base64 인코딩
        
        Args:
            frame: 입력 프레임 (BGR 또는 RGB)
            quality: JPEG 품질 (1-100)
            convert_to_rgb: 사용자 전달용 RGB로 변환 여부
        
        Returns:
            (base64_string, frame_size)
        """
        # 사용자 전달용 RGB 변환
        if convert_to_rgb and len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        compressed_data, size = self.compress_frame(frame, quality)
        base64_string = base64.b64encode(compressed_data).decode('utf-8')
        
        # 대역폭 통계 업데이트
        self._update_bandwidth_stats(size)
        
        return base64_string, size
    
    def auto_adjust_quality(self, frame: np.ndarray, convert_to_rgb: bool = False) -> Tuple[str, int, int]:
        """
        자동 품질 조절로 프레임 인코딩
        
        Args:
            frame: 입력 프레임
            convert_to_rgb: 사용자 전달용 RGB로 변환 여부
        
        Returns:
            (base64_string, frame_size, used_quality)
        """
        if not self.auto_quality_adjust:
            return (*self.encode_frame_base64(frame, self.current_quality, convert_to_rgb), self.current_quality)
        
        # 초기 품질로 테스트
        test_quality = self.current_quality
        base64_string, size = self.encode_frame_base64(frame, test_quality, convert_to_rgb)
        
        # 크기가 너무 크면 품질 낮추기
        while size > self.max_frame_size and test_quality > 20:
            test_quality -= 10
            base64_string, size = self.encode_frame_base64(frame, test_quality, convert_to_rgb)
        
        # 크기가 너무 작으면 품질 높이기 (대역폭 여유가 있을 때)
        while size < self.max_frame_size * 0.7 and test_quality < 80:
            test_quality += 10
            test_base64, test_size = self.encode_frame_base64(frame, test_quality, convert_to_rgb)
            if test_size <= self.max_frame_size:
                base64_string, size = test_base64, test_size
            else:
                break
        
        # 현재 품질 업데이트
        self.current_quality = test_quality
        self.quality_history.append(test_quality)
        
        # 히스토리 제한 (최근 100개만 유지)
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]
        
        return base64_string, size, test_quality
    
    def _update_bandwidth_stats(self, frame_size: int):
        """대역폭 통계 업데이트"""
        stats = self.bandwidth_stats
        stats['last_frame_size'] = frame_size
        stats['total_frames'] += 1
        
        # 이동 평균 계산
        if stats['avg_frame_size'] == 0:
            stats['avg_frame_size'] = frame_size
        else:
            alpha = 0.1  # 가중치
            stats['avg_frame_size'] = (alpha * frame_size + 
                                     (1 - alpha) * stats['avg_frame_size'])
    
    def get_bandwidth_stats(self) -> dict:
        """대역폭 통계 반환"""
        stats = self.bandwidth_stats.copy()
        stats['current_quality'] = self.current_quality
        stats['avg_quality'] = (sum(self.quality_history) / len(self.quality_history) 
                               if self.quality_history else self.current_quality)
        return stats
    
    def create_frame_metadata(self, frame: np.ndarray, camera_id: str, gh_idx: int) -> dict:
        """프레임 메타데이터 생성"""
        return {
            'timestamp': time.time(),
            'camera_id': camera_id,
            'gh_idx': gh_idx,
            'frame_shape': frame.shape,
            'quality': self.current_quality,
            'frame_size': self.bandwidth_stats['last_frame_size']
        }