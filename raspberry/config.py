"""
라즈베리파이 카메라 설정
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CameraConfig:
    # 카메라 식별
    camera_id: str = "cam_001"  # 각 라즈베리파이마다 고유 ID
    gh_idx: int = 74  # 온실 인덱스
    
    # 네트워크 설정
    server_host: str = "192.168.219.47"  # ML API 서버 IP
    server_port: int = 8003
    websocket_endpoint: str = "/ws/camera"
    
    # 카메라 해상도 설정
    lq_resolution: Tuple[int, int] = (320, 240)    # Low Quality (움직임 감지용)
    hq_resolution: Tuple[int, int] = (1024, 768)   # High Quality (탐지용)
    
    # 프레임 설정
    lq_fps: int = 10  # LQ 스트림 FPS
    hq_fps: int = 5   # HQ 스트림 FPS (탐지 시에만)
    
    # 움직임 감지 설정
    motion_threshold: int = 5000  # 움직임 감지 임계값
    motion_blur_size: int = 21    # 가우시안 블러 크기
    motion_min_area: int = 1000   # 최소 움직임 영역
    
    # 녹화 설정
    recording_duration: int = 10  # 녹화 지속 시간 (초)
    recording_cooldown: int = 30  # 녹화 쿨다운 (초)
    
    # 압축 설정
    jpeg_quality_low: int = 30    # 낮은 품질 (대역폭 절약)
    jpeg_quality_medium: int = 60 # 중간 품질
    jpeg_quality_high: int = 80   # 높은 품질 (탐지용)
    
    # 네트워크 대역폭 조절
    max_frame_size: int = 50 * 1024  # 최대 프레임 크기 (50KB)
    auto_quality_adjust: bool = True  # 자동 품질 조절
    
    # 로컬 저장 설정
    save_local_backup: bool = True
    local_backup_dir: str = "/home/pi/camera_backup"
    max_backup_days: int = 7  # 백업 보관 기간

def get_config() -> CameraConfig:
    """환경 변수를 고려한 설정 반환"""
    config = CameraConfig()
    
    # 환경 변수에서 설정 오버라이드
    config.camera_id = os.getenv("CAMERA_ID", config.camera_id)
    config.server_host = os.getenv("SERVER_HOST", config.server_host)
    config.gh_idx = int(os.getenv("GH_IDX", config.gh_idx))
    
    return config