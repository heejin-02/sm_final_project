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
                
                # Pi Camera v3 AF 설정 활성화
                try:
                    # 연속 자동 초점 모드 설정
                    self.camera.set_controls({
                        "AfMode": 2,  # Continuous AF
                        "AfTrigger": 0,  # Start AF
                        "LensPosition": 0.0  # 무한대 초기값
                    })
                    logger.info("✅ Pi Camera v3 AF 모드 활성화 (Continuous AF)")
                except Exception as af_error:
                    logger.warning(f"⚠️ AF 설정 실패 (v2 카메라이거나 AF 미지원): {af_error}")
                
                self.camera.start()
                time.sleep(3)  # 카메라 및 AF 안정화 대기 (AF 때문에 시간 증가)
                
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
    
    def trigger_autofocus(self):
        """수동 AF 트리거 (필요시 초점 재조정)"""
        if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
            try:
                self.camera.set_controls({"AfTrigger": 1})  # AF 트리거
                logger.debug("AF 수동 트리거 실행")
            except Exception as e:
                logger.warning(f"AF 트리거 실패: {e}")
    
    def set_focus_mode(self, mode: str = "continuous"):
        """AF 모드 설정
        
        Args:
            mode: "manual", "auto", "continuous"
        """
        if not (RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera):
            return
            
        mode_map = {
            "manual": 0,      # Manual focus
            "auto": 1,        # Auto focus (single shot)
            "continuous": 2   # Continuous auto focus
        }
        
        try:
            af_mode = mode_map.get(mode, 2)
            self.camera.set_controls({"AfMode": af_mode})
            logger.info(f"AF 모드 변경: {mode} (값: {af_mode})")
        except Exception as e:
            logger.warning(f"AF 모드 설정 실패: {e}")
    
    def set_focus_distance(self, distance: float):
        """수동 초점 거리 설정 (0.0 = 무한대, 10.0 = 가까이)"""
        if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
            try:
                self.camera.set_controls({
                    "AfMode": 0,  # Manual mode
                    "LensPosition": distance
                })
                logger.info(f"수동 초점 거리 설정: {distance}")
            except Exception as e:
                logger.warning(f"초점 거리 설정 실패: {e}")

    def capture_frame(self, high_quality: bool = False, color_format: str = "BGR") -> Optional[np.ndarray]:
        """프레임 캡처
        
        Args:
            high_quality: 고화질 모드 여부
            color_format: 출력 색상 포맷 ("BGR", "RGB")
        """
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
                    
                    # 색상 채널 변환 처리
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Picamera2는 RGB로 출력
                        if color_format == "BGR":
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        # RGB 요청이면 그대로 유지
                    elif len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        if color_format == "RGB":
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    return frame
            
            elif self.cap:
                # OpenCV 웹캠 사용 (기본 BGR)
                ret, frame = self.cap.read()
                if ret:
                    if not high_quality:
                        frame = cv2.resize(frame, self.lq_resolution)
                    
                    # BGR에서 RGB로 변환 (필요시)
                    if color_format == "RGB":
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    return frame
            
            return None
            
        except Exception as e:
            logger.error(f"프레임 캡처 오류: {e}")
            return None
    
    def get_camera_type(self) -> str:
        """카메라 타입 반환"""
        return "picamera2" if PICAMERA2_AVAILABLE else "opencv"
    
    def capture_frame_for_ml(self, high_quality: bool = True) -> Optional[np.ndarray]:
        """ML 모델 입력용 프레임 캡처 (BGR 고정)"""
        return self.capture_frame(high_quality=high_quality, color_format="BGR")
    
    def capture_frame_for_user(self, high_quality: bool = False) -> Optional[np.ndarray]:
        """사용자 전달용 프레임 캡처 (RGB)"""
        return self.capture_frame(high_quality=high_quality, color_format="RGB")

    def get_focus_info(self) -> dict:
        """현재 AF 상태 정보 반환"""
        info = {
            "af_available": False,
            "af_mode": "unknown",
            "af_state": "unknown", 
            "lens_position": 0.0
        }
        
        if RASPBERRY_PI and PICAMERA2_AVAILABLE and self.camera:
            try:
                metadata = self.camera.capture_metadata()
                info.update({
                    "af_available": True,
                    "af_mode": metadata.get("AfMode", "unknown"),
                    "af_state": metadata.get("AfState", "unknown"),
                    "lens_position": metadata.get("LensPosition", 0.0)
                })
            except Exception as e:
                logger.warning(f"AF 정보 조회 실패: {e}")
        
        return info
    
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