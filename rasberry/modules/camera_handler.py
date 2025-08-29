"""
카메라 핸들러 모듈 - Picamera2/WebCam 추상화
"""

import cv2
import numpy as np
import time
import logging
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Picamera2 가용성 확인
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


class CameraHandler:
    """카메라 핸들러 - Picamera2와 OpenCV WebCam을 추상화"""
    
    def __init__(self, lq_resolution: Tuple[int, int], hq_resolution: Tuple[int, int]):
        self.lq_resolution = lq_resolution
        self.hq_resolution = hq_resolution
        self.camera = None
        self.cap = None
        
        self._init_camera()
    
    def _init_camera(self):
        """카메라 초기화 - Bookworm 호환"""
        global RASPBERRY_PI, PICAMERA2_AVAILABLE
        
        if RASPBERRY_PI and PICAMERA2_AVAILABLE:
            try:
                logger.info("Picamera2 초기화 시도 (Bookworm)...")
                
                self.camera = Picamera2()
                
                logger.info(f"사용 가능한 센서 모드: {len(self.camera.sensor_modes)}개")
                
                # 듀얼 스트림 설정 시도
                try:
                    config = self.camera.create_preview_configuration(
                        main={
                            "size": self.hq_resolution,
                            "format": "RGB888"
                        },
                        lores={
                            "size": self.lq_resolution,
                            "format": "YUV420"
                        },
                        buffer_count=2
                    )
                    logger.info(f"듀얼 스트림 설정: HQ={self.hq_resolution}, LQ={self.lq_resolution}")
                except Exception as e:
                    logger.warning(f"듀얼 스트림 설정 실패: {e}")
                    # 단일 스트림으로 폴백
                    config = self.camera.create_preview_configuration(
                        main={
                            "size": self.hq_resolution,
                            "format": "RGB888"
                        }
                    )
                    logger.info(f"단일 스트림 설정: {self.hq_resolution}")
                
                self.camera.configure(config)
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
                
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
        
        # Picamera2를 사용할 수 없으면 OpenCV 웹캠 사용
        if not RASPBERRY_PI or not PICAMERA2_AVAILABLE:
            logger.info("OpenCV 웹캠 초기화 시도...")
            
            for i in range(5):
                try:
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.hq_resolution[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.hq_resolution[1])
                            logger.info(f"✅ 웹캠 초기화 성공 (인덱스 {i})")
                            break
                    self.cap.release()
                except:
                    pass
            else:
                logger.error("❌ 카메라를 찾을 수 없습니다!")
                raise RuntimeError("No camera available")
    
    def capture_frame(self, high_quality: bool = False) -> Optional[np.ndarray]:
        """프레임 캡처"""
        try:
            if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
                # Picamera2 사용
                if hasattr(self.camera, 'capture_array'):
                    if not high_quality:
                        try:
                            frame = self.camera.capture_array("lores")
                        except:
                            frame = self.camera.capture_array("main")
                            frame = cv2.resize(frame, self.lq_resolution)
                    else:
                        frame = self.camera.capture_array("main")
                    
                    # RGB를 BGR로 변환 (OpenCV 호환성)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    return frame
            
            elif self.cap:
                # OpenCV 웹캠 사용
                ret, frame = self.cap.read()
                if ret:
                    if not high_quality:
                        frame = cv2.resize(frame, self.lq_resolution)
                    return frame
            
            return None
            
        except Exception as e:
            logger.error(f"프레임 캡처 오류: {e}")
            return None
    
    def get_camera_type(self) -> str:
        """카메라 타입 반환"""
        return "picamera2" if PICAMERA2_AVAILABLE else "opencv"
    
    def cleanup(self):
        """카메라 정리"""
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