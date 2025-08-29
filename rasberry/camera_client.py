"""
라즈베리파이 카메라 클라이언트 - 모듈화된 버전
실시간 웹소켓을 통한 영상 전송 및 움직임 감지
"""

import asyncio
import json
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import sys
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 로컬 모듈 import
try:
    from config import get_config
    from utils.motion_detector import MotionDetector
    from utils.frame_processor import FrameProcessor
    from modules.camera_handler import CameraHandler
    from modules.websocket_client import WebSocketClient
except ImportError as e:
    logger.error(f"로컬 모듈 import 실패: {e}")
    sys.exit(1)

class CameraClient:
    """카메라 클라이언트 - 모듈화된 구조"""
    
    def __init__(self):
        self.config = get_config()
        
        # 모듈 초기화
        self.camera_handler = CameraHandler(
            self.config.lq_resolution,
            self.config.hq_resolution
        )
        
        self.motion_detector = MotionDetector(
            threshold=self.config.motion_threshold,
            blur_size=self.config.motion_blur_size,
            min_area=self.config.motion_min_area
        )
        
        self.frame_processor = FrameProcessor(
            max_frame_size=self.config.max_frame_size,
            auto_quality_adjust=self.config.auto_quality_adjust
        )
        
        self.websocket_client = WebSocketClient(
            self.config.server_host,
            self.config.server_port,
            self.config.websocket_endpoint,
            self.config.camera_id,
            self.config.gh_idx
        )
        
        # 상태 관리
        self.is_recording = False
        self.recording_start_time = 0
        self.last_detection_time = 0
        
        # 영상 버퍼
        self.lq_buffer = []
        self.hq_buffer = []
        self.frame_timestamps = []
        
        # 로컬 백업 설정
        if self.config.save_local_backup:
            Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)
    
    async def process_motion_detection(self, lq_frame: np.ndarray):
        """움직임 감지 처리"""
        motion_detected, motion_areas = self.motion_detector.detect_motion(lq_frame)
        
        current_time = time.time()
        
        if motion_detected:
            logger.info(f"🚨 움직임 감지! 영역 수: {len(motion_areas)}")
            
            await self.websocket_client.send_recording_event(
                "motion_detected",
                motion_areas=[[int(x), int(y), int(w), int(h)] for x, y, w, h in motion_areas]
            )
            
            # 녹화 시작 조건 확인
            if not self.is_recording and (current_time - self.last_detection_time) > self.config.recording_cooldown:
                self.is_recording = True
                self.recording_start_time = current_time
                self.last_detection_time = current_time
                
                logger.info(f"🔴 녹화 시작 - 지속시간: {self.config.recording_duration}초")
                await self.websocket_client.send_recording_event("recording_start")
        
        return motion_detected, motion_areas
    
    async def add_frame_to_buffer(self, lq_frame: np.ndarray, hq_frame: np.ndarray, timestamp: float):
        """프레임을 버퍼에 추가"""
        if not self.is_recording:
            return
        
        self.lq_buffer.append(lq_frame.copy())
        self.hq_buffer.append(hq_frame.copy())
        self.frame_timestamps.append(timestamp)
        
        # 메모리 관리
        max_frames = int(self.config.lq_fps * 15)
        if len(self.lq_buffer) > max_frames:
            self.lq_buffer.pop(0)
            self.hq_buffer.pop(0)
            self.frame_timestamps.pop(0)
        
        # 녹화 종료 조건 확인
        if time.time() - self.recording_start_time > self.config.recording_duration:
            await self.finish_recording()
    
    async def finish_recording(self):
        """녹화 종료 및 버퍼 데이터 전송"""
        self.is_recording = False
        logger.info("⏹️ 녹화 종료 - 버퍼 데이터 전송 시작")
        
        if len(self.lq_buffer) > 0 and len(self.hq_buffer) > 0:
            await self.send_video_buffer()
        
        # 버퍼 클리어
        self.lq_buffer.clear()
        self.hq_buffer.clear() 
        self.frame_timestamps.clear()
        
        await self.websocket_client.send_recording_event("recording_stop")
    
    async def send_video_buffer(self):
        """녹화된 영상 데이터를 서버로 전송"""
        logger.info(f"🎦 비디오 버퍼 전송: LQ {len(self.lq_buffer)}프레임, HQ {len(self.hq_buffer)}프레임")
        
        try:
            # 프레임들을 Base64로 인코딩
            lq_frames_b64 = []
            hq_frames_b64 = []
            
            for i, (lq_frame, hq_frame) in enumerate(zip(self.lq_buffer, self.hq_buffer)):
                lq_b64, _ = self.frame_processor.encode_frame_base64(lq_frame, quality=30)
                hq_b64, _ = self.frame_processor.encode_frame_base64(hq_frame, quality=50) 
                
                lq_frames_b64.append(lq_b64)
                hq_frames_b64.append(hq_b64)
                
                if i % 20 == 0:
                    logger.info(f"인코딩 진행: {i+1}/{len(self.lq_buffer)}")
            
            # 비디오 버퍼 전송
            await self.websocket_client.send_video_buffer(
                lq_frames_b64, hq_frames_b64, self.frame_timestamps,
                self.recording_start_time, self.config.recording_duration,
                self.config.lq_resolution, self.config.hq_resolution
            )
            
        except Exception as e:
            logger.error(f"❌ 비디오 버퍼 전송 실패: {e}")
    
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
        logger.info("📹 카메라 루프 시작")
        
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # LQ 프레임 캡처 (ML용 BGR)
                lq_frame = self.camera_handler.capture_frame_for_ml(high_quality=False)
                if lq_frame is None:
                    error_count += 1
                    if error_count > max_errors:
                        logger.error(f"프레임 캡처 실패 {max_errors}회 초과")
                        logger.info("카메라 재초기화 시도...")
                        self.camera_handler._init_camera()
                        error_count = 0
                    await asyncio.sleep(0.1)
                    continue
                
                error_count = 0
                frame_count += 1
                
                # 움직임 감지
                motion_detected, motion_areas = await self.process_motion_detection(lq_frame)
                
                # LQ 프레임 전송
                lq_b64, lq_size, quality = self.frame_processor.auto_adjust_quality(lq_frame)
                await self.websocket_client.send_frame_data(
                    lq_b64, "lq", motion_detected,
                    [[int(x), int(y), int(w), int(h)] for x, y, w, h in motion_areas],
                    lq_size, quality, lq_frame.shape
                )
                
                # HQ 프레임 캡처 (ML용 BGR)
                hq_frame = self.camera_handler.capture_frame_for_ml(high_quality=True)
                if hq_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # 녹화 중이면 프레임을 버퍼에 저장
                if self.is_recording:
                    current_timestamp = time.time()
                    await self.add_frame_to_buffer(lq_frame, hq_frame, current_timestamp)
                    
                    # 로컬 백업 저장 (선택사항)
                    if frame_count % 10 == 0:
                        await self.save_local_backup(hq_frame, "hq")
                
                # 실시간 상태 전송
                if motion_detected:
                    await self.websocket_client.send_recording_event(
                        "motion_frame",
                        motion_areas=[[int(x), int(y), int(w), int(h)] for x, y, w, h in motion_areas],
                        timestamp=time.time()
                    )
                
                # 주기적 상태 로깅
                if frame_count % 100 == 0:
                    logger.info(f"📊 처리된 프레임: {frame_count}, 녹화 중: {self.is_recording}")
                
                # FPS 제어
                await asyncio.sleep(1.0 / self.config.lq_fps)
                
            except Exception as e:
                logger.error(f"카메라 루프 오류: {e}")
                error_count += 1
                await asyncio.sleep(1)
    
    async def websocket_handler(self):
        """웹소켓 연결 및 유지"""
        retry_count = 0
        max_retries = 5
        
        while True:
            try:
                # 웹소켓 연결
                if await self.websocket_client.connect():
                    retry_count = 0
                    
                    # 초기화 메시지에 카메라 타입 포함
                    await self.websocket_client.send_init_message(
                        self.camera_handler.get_camera_type()
                    )
                    
                    # 카메라 루프 실행
                    await self.camera_loop()
                else:
                    raise Exception("웹소켓 연결 실패")
                    
            except Exception as e:
                logger.error(f"❌ 웹소켓 연결 오류: {e}")
                
                retry_count += 1
                if retry_count > max_retries:
                    wait_time = 30
                else:
                    wait_time = min(5 * retry_count, 30)
                
                logger.info(f"⏰ {wait_time}초 후 재연결 시도... (시도 {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
    
    def cleanup(self):
        """정리 작업"""
        logger.info("🧹 정리 작업 시작...")
        self.camera_handler.cleanup()
        logger.info("✅ 카메라 클라이언트 종료")

async def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("라즈베리파이 카메라 클라이언트 시작 (모듈화 버전)")
    logger.info(f"Python 버전: {sys.version}")
    logger.info("=" * 60)
    
    client = CameraClient()
    
    try:
        await client.websocket_handler()
    except KeyboardInterrupt:
        logger.info("⛔ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
    finally:
        client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)