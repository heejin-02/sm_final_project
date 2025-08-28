"""
라즈베리파이 카메라 클라이언트
실시간 웹소켓을 통한 영상 전송 및 움직임 감지
"""

import asyncio
import websockets
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FileOutput
    RASPBERRY_PI = True
except ImportError:
    print("PiCamera2 없음 - 웹캠으로 대체")
    RASPBERRY_PI = False

from config import get_config
from utils.motion_detector import MotionDetector
from utils.frame_processor import FrameProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraClient:
    def __init__(self):
        self.config = get_config()
        self.motion_detector = MotionDetector(
            threshold=self.config.motion_threshold,
            blur_size=self.config.motion_blur_size,
            min_area=self.config.motion_min_area
        )
        self.frame_processor = FrameProcessor(
            max_frame_size=self.config.max_frame_size,
            auto_quality_adjust=self.config.auto_quality_adjust
        )
        
        # 상태 관리
        self.is_recording = False
        self.recording_start_time = 0
        self.last_detection_time = 0
        self.websocket = None
        
        # 카메라 초기화
        self.camera = None
        self.cap = None
        self._init_camera()
        
        # 로컬 백업 설정
        if self.config.save_local_backup:
            Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_camera(self):
        """카메라 초기화"""
        if RASPBERRY_PI:
            self.camera = Picamera2()
            # LQ와 HQ 스트림 설정
            config = self.camera.create_preview_configuration(
                main={
                    "size": self.config.hq_resolution, 
                    "format": "RGB888"
                },
                lores={
                    "size": self.config.lq_resolution,
                    "format": "RGB888" 
                }
            )
            self.camera.configure(config)
            self.camera.start()
            logger.info(f"PiCamera 초기화 완료 - HQ: {self.config.hq_resolution}, LQ: {self.config.lq_resolution}")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.hq_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.hq_resolution[1])
            logger.info("웹캠 초기화 완료")
    
    def capture_frame(self, high_quality: bool = False) -> Optional[np.ndarray]:
        """프레임 캡처"""
        if RASPBERRY_PI:
            if high_quality:
                return self.camera.capture_array("main")
            else:
                return self.camera.capture_array("lores")
        else:
            ret, frame = self.cap.read()
            if ret:
                if not high_quality:
                    frame = cv2.resize(frame, self.config.lq_resolution)
                return frame
            return None
    
    async def send_frame_data(self, frame: np.ndarray, frame_type: str, motion_detected: bool = False, motion_areas: list = None):
        """웹소켓으로 프레임 데이터 전송"""
        if not self.websocket:
            return
        
        try:
            # 프레임 인코딩
            if frame_type == "lq":
                base64_frame, size, quality = self.frame_processor.auto_adjust_quality(frame)
            else:
                base64_frame, size = self.frame_processor.encode_frame_base64(
                    frame, self.config.jpeg_quality_high
                )
                quality = self.config.jpeg_quality_high
            
            # 메시지 구성
            message = {
                "type": "frame_data",
                "camera_id": self.config.camera_id,
                "gh_idx": self.config.gh_idx,
                "frame_type": frame_type,
                "timestamp": time.time(),
                "motion_detected": motion_detected,
                "motion_areas": motion_areas or [],
                "frame_data": base64_frame,
                "frame_size": size,
                "quality": quality,
                "frame_shape": frame.shape
            }
            
            await self.websocket.send(json.dumps(message))
            
            # 통계 로깅
            if frame_type == "lq":
                stats = self.frame_processor.get_bandwidth_stats()
                logger.debug(f"프레임 전송: {frame_type} | 크기: {size}B | 품질: {quality} | 평균 크기: {stats['avg_frame_size']:.0f}B")
            
        except Exception as e:
            logger.error(f"프레임 전송 오류: {e}")
    
    async def send_recording_event(self, event_type: str, **kwargs):
        """녹화 이벤트 전송"""
        if not self.websocket:
            return
        
        try:
            message = {
                "type": "recording_event",
                "camera_id": self.config.camera_id,
                "gh_idx": self.config.gh_idx,
                "event_type": event_type,  # start, stop, motion_detected
                "timestamp": time.time(),
                **kwargs
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"녹화 이벤트 전송: {event_type}")
            
        except Exception as e:
            logger.error(f"녹화 이벤트 전송 오류: {e}")
    
    async def process_motion_detection(self, lq_frame: np.ndarray):
        """움직임 감지 처리"""
        motion_detected, motion_areas = self.motion_detector.detect_motion(lq_frame)
        
        current_time = time.time()
        
        if motion_detected:
            logger.info(f"움직임 감지! 영역 수: {len(motion_areas)}")
            
            # 움직임 감지 이벤트 전송
            await self.send_recording_event(
                "motion_detected",
                motion_areas=[[int(x), int(y), int(w), int(h)] for x, y, w, h in motion_areas]
            )
            
            # 녹화 시작 조건 확인
            if not self.is_recording and (current_time - self.last_detection_time) > self.config.recording_cooldown:
                self.is_recording = True
                self.recording_start_time = current_time
                self.last_detection_time = current_time
                
                logger.info(f"녹화 시작 - 지속시간: {self.config.recording_duration}초")
                await self.send_recording_event("recording_start")
        
        return motion_detected, motion_areas
    
    async def process_recording(self, hq_frame: np.ndarray):
        """녹화 중 HQ 프레임 처리"""
        if not self.is_recording:
            return
        
        # HQ 프레임 전송 (탐지용)
        await self.send_frame_data(hq_frame, "hq", motion_detected=True)
        
        # 녹화 종료 조건 확인
        if time.time() - self.recording_start_time > self.config.recording_duration:
            self.is_recording = False
            logger.info("녹화 종료")
            await self.send_recording_event("recording_stop")
    
    async def save_local_backup(self, frame: np.ndarray, frame_type: str):
        """로컬 백업 저장"""
        if not self.config.save_local_backup:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.config.camera_id}_{frame_type}_{timestamp}.jpg"
            filepath = Path(self.config.local_backup_dir) / filename
            
            cv2.imwrite(str(filepath), frame)
            
        except Exception as e:
            logger.error(f"로컬 백업 저장 오류: {e}")
    
    async def camera_loop(self):
        """메인 카메라 루프"""
        logger.info("카메라 루프 시작")
        
        while True:
            try:
                # LQ 프레임 캡처 (항상)
                lq_frame = self.capture_frame(high_quality=False)
                if lq_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # 움직임 감지
                motion_detected, motion_areas = await self.process_motion_detection(lq_frame)
                
                # LQ 프레임 전송
                await self.send_frame_data(lq_frame, "lq", motion_detected, 
                                         [[int(x), int(y), int(w), int(h)] for x, y, w, h in motion_areas])
                
                # 녹화 중이면 HQ 프레임도 처리
                if self.is_recording:
                    hq_frame = self.capture_frame(high_quality=True)
                    if hq_frame is not None:
                        await self.process_recording(hq_frame)
                        
                        # 로컬 백업 저장
                        await self.save_local_backup(hq_frame, "hq")
                
                # FPS 제어
                await asyncio.sleep(1.0 / self.config.lq_fps)
                
            except Exception as e:
                logger.error(f"카메라 루프 오류: {e}")
                await asyncio.sleep(1)
    
    async def websocket_handler(self):
        """웹소켓 연결 및 유지"""
        uri = f"ws://{self.config.server_host}:{self.config.server_port}{self.config.websocket_endpoint}"
        
        while True:
            try:
                logger.info(f"웹소켓 연결 시도: {uri}")
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    logger.info("웹소켓 연결 성공")
                    
                    # 초기 연결 메시지 전송
                    init_message = {
                        "type": "camera_init",
                        "camera_id": self.config.camera_id,
                        "gh_idx": self.config.gh_idx,
                        "config": {
                            "lq_resolution": self.config.lq_resolution,
                            "hq_resolution": self.config.hq_resolution,
                            "lq_fps": self.config.lq_fps
                        }
                    }
                    await websocket.send(json.dumps(init_message))
                    
                    # 카메라 루프 실행
                    await self.camera_loop()
                    
            except Exception as e:
                logger.error(f"웹소켓 연결 오류: {e}")
                self.websocket = None
                await asyncio.sleep(5)  # 5초 후 재연결 시도
    
    def cleanup(self):
        """정리 작업"""
        if RASPBERRY_PI and self.camera:
            self.camera.stop()
        elif self.cap:
            self.cap.release()
        
        logger.info("카메라 클라이언트 종료")

async def main():
    """메인 함수"""
    client = CameraClient()
    
    try:
        await client.websocket_handler()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    finally:
        client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())