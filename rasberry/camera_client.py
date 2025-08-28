"""
라즈베리파이 카메라 클라이언트 - Bookworm OS 버전
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
import sys
import subprocess
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Picamera2 import 시도
RASPBERRY_PI = False
PICAMERA2_AVAILABLE = False

def check_rpicam_system():
    """rpicam 시스템 확인 (Bookworm)"""
    try:
        result = subprocess.run(['rpicam-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=5)
        if "No cameras available" not in result.stdout and result.returncode == 0:
            logger.info("rpicam 카메라 시스템 확인됨")
            return True
    except:
        pass
    return False

# Bookworm OS에서 Picamera2 import
try:
    # 먼저 시스템 카메라 확인
    if check_rpicam_system():
        from picamera2 import Picamera2
        RASPBERRY_PI = True
        PICAMERA2_AVAILABLE = True
        logger.info("✅ Picamera2 모듈 로드 성공 (Bookworm)")
except ImportError as e:
    logger.warning(f"⚠️ Picamera2 import 실패: {e}")
    logger.warning("웹캠으로 대체합니다")
except Exception as e:
    logger.error(f"❌ Picamera2 초기화 오류: {e}")

# 로컬 모듈 import
try:
    from config import get_config
    from utils.motion_detector import MotionDetector
    from utils.frame_processor import FrameProcessor
except ImportError:
    logger.warning("로컬 모듈 import 실패, 기본 설정 사용")
    # 기본 설정 클래스 정의
    class DefaultConfig:
        camera_id = "cam_001"
        gh_idx = 74
        server_host = "192.168.219.47"  # Spring Boot 서버 IP
        server_port = 8095
        websocket_endpoint = "/api/camera/websocket"
        lq_resolution = (320, 240)
        hq_resolution = (1024, 768)
        lq_fps = 10
        hq_fps = 5
        motion_threshold = 5000
        motion_blur_size = 21
        motion_min_area = 1000
        recording_duration = 10
        recording_cooldown = 30
        jpeg_quality_low = 30
        jpeg_quality_medium = 60
        jpeg_quality_high = 80
        max_frame_size = 50 * 1024
        auto_quality_adjust = True
        save_local_backup = True
        local_backup_dir = "/home/pi/camera_backup"
        max_backup_days = 7
    
    def get_config():
        return DefaultConfig()
    
    # 간단한 MotionDetector 구현
    class MotionDetector:
        def __init__(self, threshold=5000, blur_size=21, min_area=1000):
            self.threshold = threshold
            self.blur_size = blur_size
            self.min_area = min_area
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50, history=500
            )
            
        def detect_motion(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            fg_mask = self.background_subtractor.apply(blurred)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            bounding_boxes = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
                    motion_detected = True
            
            return motion_detected, bounding_boxes
    
    # 간단한 FrameProcessor 구현
    class FrameProcessor:
        def __init__(self, max_frame_size=50*1024, auto_quality_adjust=True):
            self.max_frame_size = max_frame_size
            self.auto_quality_adjust = auto_quality_adjust
            self.current_quality = 60
            
        def auto_adjust_quality(self, frame):
            import base64
            quality = self.current_quality
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            compressed_data = buffer.tobytes()
            base64_string = base64.b64encode(compressed_data).decode('utf-8')
            size = len(compressed_data)
            return base64_string, size, quality
        
        def encode_frame_base64(self, frame, quality=60):
            import base64
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            compressed_data = buffer.tobytes()
            base64_string = base64.b64encode(compressed_data).decode('utf-8')
            return base64_string, len(compressed_data)
        
        def get_bandwidth_stats(self):
            return {'avg_frame_size': 0}

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
        """카메라 초기화 - Bookworm 호환"""
        global RASPBERRY_PI, PICAMERA2_AVAILABLE
        
        if RASPBERRY_PI and PICAMERA2_AVAILABLE:
            try:
                logger.info("Picamera2 초기화 시도 (Bookworm)...")
                
                # Picamera2 인스턴스 생성
                self.camera = Picamera2()
                
                # 사용 가능한 카메라 모드 확인
                logger.info(f"사용 가능한 센서 모드: {len(self.camera.sensor_modes)}개")
                
                # Camera Module v3용 설정 생성
                try:
                    # 듀얼 스트림 설정 시도 (main + lores)
                    config = self.camera.create_preview_configuration(
                        main={
                            "size": self.config.hq_resolution,
                            "format": "RGB888"
                        },
                        lores={
                            "size": self.config.lq_resolution,
                            "format": "YUV420"
                        },
                        buffer_count=2  # 버퍼 수 조정
                    )
                    logger.info(f"듀얼 스트림 설정: HQ={self.config.hq_resolution}, LQ={self.config.lq_resolution}")
                except Exception as e:
                    logger.warning(f"듀얼 스트림 설정 실패: {e}")
                    # 단일 스트림으로 폴백
                    config = self.camera.create_preview_configuration(
                        main={
                            "size": self.config.hq_resolution,
                            "format": "RGB888"
                        }
                    )
                    logger.info(f"단일 스트림 설정: {self.config.hq_resolution}")
                
                # 설정 적용
                self.camera.configure(config)
                
                # 카메라 시작
                self.camera.start()
                time.sleep(2)  # 카메라 안정화 대기
                
                # 테스트 캡처
                test_frame = self.camera.capture_array("main")
                logger.info(f"✅ Picamera2 초기화 성공! 테스트 프레임: {test_frame.shape}")
                
            except Exception as e:
                logger.error(f"❌ Picamera2 초기화 실패: {e}")
                logger.warning("웹캠으로 폴백 시도...")
                RASPBERRY_PI = False
                PICAMERA2_AVAILABLE = False
                
                # Picamera2 실패 시 정리
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
        
        # Picamera2를 사용할 수 없으면 OpenCV 웹캠 사용
        if not RASPBERRY_PI or not PICAMERA2_AVAILABLE:
            logger.info("OpenCV 웹캠 초기화 시도...")
            
            # 여러 카메라 인덱스 시도
            for i in range(5):
                try:
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.hq_resolution[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.hq_resolution[1])
                            logger.info(f"✅ 웹캠 초기화 성공 (인덱스 {i})")
                            break
                    self.cap.release()
                except:
                    pass
            else:
                logger.error("❌ 카메라를 찾을 수 없습니다!")
                raise RuntimeError("No camera available")
    
    def capture_frame(self, high_quality: bool = False) -> Optional[np.ndarray]:
        """프레임 캡처 - Bookworm 호환"""
        try:
            if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
                # Picamera2 사용
                if hasattr(self.camera, 'capture_array'):
                    # lores 스트림이 있는지 확인
                    if not high_quality:
                        try:
                            # lores 스트림 캡처 시도
                            frame = self.camera.capture_array("lores")
                        except:
                            # lores가 없으면 main에서 캡처 후 리사이즈
                            frame = self.camera.capture_array("main")
                            frame = cv2.resize(frame, self.config.lq_resolution)
                    else:
                        frame = self.camera.capture_array("main")
                    
                    # RGB를 BGR로 변환 (OpenCV 호환성)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 2:
                        # 그레이스케일을 3채널 BGR로 변환
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    return frame
            
            elif self.cap:
                # OpenCV 웹캠 사용
                ret, frame = self.cap.read()
                if ret:
                    if not high_quality:
                        frame = cv2.resize(frame, self.config.lq_resolution)
                    return frame
            
            return None
            
        except Exception as e:
            logger.error(f"프레임 캡처 오류: {e}")
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
                logger.debug(f"프레임 전송: {frame_type} | 크기: {size}B | 품질: {quality}")
            
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
                "event_type": event_type,
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
            logger.info(f"🚨 움직임 감지! 영역 수: {len(motion_areas)}")
            
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
                
                logger.info(f"🔴 녹화 시작 - 지속시간: {self.config.recording_duration}초")
                await self.send_recording_event("recording_start")
        
        return motion_detected, motion_areas
    
    async def process_recording(self, hq_frame: np.ndarray):
        """녹화 중 HQ 프레임 처리"""
        if not self.is_recording:
            return
        
        # HQ 프레임 전송
        await self.send_frame_data(hq_frame, "hq", motion_detected=True)
        
        # 녹화 종료 조건 확인
        if time.time() - self.recording_start_time > self.config.recording_duration:
            self.is_recording = False
            logger.info("⏹️ 녹화 종료")
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
        logger.info("📹 카메라 루프 시작")
        
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # LQ 프레임 캡처
                lq_frame = self.capture_frame(high_quality=False)
                if lq_frame is None:
                    error_count += 1
                    if error_count > max_errors:
                        logger.error(f"프레임 캡처 실패 {max_errors}회 초과")
                        # 카메라 재초기화 시도
                        logger.info("카메라 재초기화 시도...")
                        self._init_camera()
                        error_count = 0
                    await asyncio.sleep(0.1)
                    continue
                
                error_count = 0  # 성공 시 에러 카운트 리셋
                frame_count += 1
                
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
        uri = f"ws://{self.config.server_host}:{self.config.server_port}{self.config.websocket_endpoint}"
        
        retry_count = 0
        max_retries = 5
        
        while True:
            try:
                logger.info(f"🔌 웹소켓 연결 시도: {uri}")
                async with websockets.connect(uri) as websocket:
                    self.websocket = websocket
                    logger.info("✅ 웹소켓 연결 성공")
                    retry_count = 0  # 연결 성공 시 재시도 카운트 리셋
                    
                    # 초기 연결 메시지 전송
                    init_message = {
                        "type": "camera_init",
                        "camera_id": self.config.camera_id,
                        "gh_idx": self.config.gh_idx,
                        "config": {
                            "lq_resolution": self.config.lq_resolution,
                            "hq_resolution": self.config.hq_resolution,
                            "lq_fps": self.config.lq_fps,
                            "camera_type": "picamera2" if PICAMERA2_AVAILABLE else "opencv"
                        }
                    }
                    await websocket.send(json.dumps(init_message))
                    logger.info("📤 초기화 메시지 전송 완료")
                    
                    # 카메라 루프 실행
                    await self.camera_loop()
                    
            except Exception as e:
                logger.error(f"❌ 웹소켓 연결 오류: {e}")
                self.websocket = None
                
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
        
        if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                logger.info("Picamera2 정리 완료")
            except:
                pass
        elif self.cap:
            try:
                self.cap.release()
                logger.info("웹캠 정리 완료")
            except:
                pass
        
        logger.info("✅ 카메라 클라이언트 종료")

async def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("라즈베리파이 카메라 클라이언트 시작")
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"Picamera2 사용 가능: {PICAMERA2_AVAILABLE}")
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
    # 시스템 Python 사용 권장 메시지
    if not PICAMERA2_AVAILABLE and RASPBERRY_PI:
        print("\n" + "=" * 60)
        print("⚠️  Picamera2를 사용할 수 없습니다!")
        print("다음 명령어로 실행해보세요:")
        print("  sudo /usr/bin/python3 camera_client.py")
        print("또는 필요한 패키지를 설치하세요:")
        print("  sudo apt install python3-picamera2")
        print("=" * 60 + "\n")
    
    # asyncio 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)