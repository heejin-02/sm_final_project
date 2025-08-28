"""
YOLO 탐지 서비스
기존 detect.py의 YOLO 로직을 서비스로 분리
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Tuple, Optional

# YOLOv5 경로 추가
YOLO_PATH = Path(__file__).parent.parent.parent / "model" / "yolov5"
sys.path.append(str(YOLO_PATH))

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes, check_img_size
    from utils.torch_utils import select_device, smart_inference_mode
    YOLO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YOLOv5 import 실패: {e}")
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class YOLOService:
    """YOLO 탐지 서비스"""
    
    def __init__(self, 
                 weights_path: str = "model/yolov5/best_clean.pt",
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 device: str = ""):
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = None
        self.device = None
        self.names = None
        self.is_initialized = False
        
        if YOLO_AVAILABLE:
            self._initialize_model(weights_path, device)
        else:
            logger.error("YOLOv5를 사용할 수 없습니다")
    
    def _initialize_model(self, weights_path: str, device: str):
        """YOLO 모델 초기화"""
        try:
            # 절대 경로로 변환
            weights_path = Path(__file__).parent.parent.parent / weights_path
            
            if not weights_path.exists():
                logger.error(f"모델 파일이 없습니다: {weights_path}")
                return
            
            # 디바이스 선택
            self.device = select_device(device)
            
            # 모델 로드
            self.model = DetectMultiBackend(weights_path, device=self.device, data=None, fp16=False)
            self.names = self.model.names
            
            # 모델 워밍업
            imgsz = check_img_size((640, 640), s=self.model.stride)
            self.model.warmup(imgsz=(1, 3, *imgsz))
            
            self.is_initialized = True
            logger.info(f"YOLO 모델 초기화 완료: {weights_path}")
            logger.info(f"클래스 목록: {self.names}")
            
        except Exception as e:
            logger.error(f"YOLO 모델 초기화 실패: {e}")
    
    @smart_inference_mode()
    async def detect_insects(self, frame: np.ndarray) -> List[Dict]:
        """
        프레임에서 해충 탐지
        
        Args:
            frame: OpenCV 프레임 (BGR)
            
        Returns:
            탐지 결과 리스트 [{"class_name": str, "confidence": float, "bbox": [x_min, y_min, x_max, y_max]}]
        """
        if not self.is_initialized:
            logger.warning("YOLO 모델이 초기화되지 않음")
            return []
        
        try:
            # 전처리
            img = self._preprocess_frame(frame)
            
            # 추론
            pred = self.model(img)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)
            
            # 후처리
            detections = self._postprocess_predictions(pred[0], img.shape[2:], frame.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO 탐지 오류: {e}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """프레임 전처리"""
        # BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 640x640으로 리사이즈
        img = cv2.resize(img, (640, 640))
        
        # numpy to tensor
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 정규화
        
        if img.ndim == 3:
            img = img[None]  # 배치 차원 추가
        
        return img
    
    def _postprocess_predictions(self, pred: torch.Tensor, 
                               model_shape: Tuple[int, int], 
                               original_shape: Tuple[int, int, int]) -> List[Dict]:
        """예측 결과 후처리"""
        detections = []
        
        if pred is not None and len(pred):
            # 좌표를 원본 이미지 크기로 변환
            pred[:, :4] = scale_boxes(model_shape, pred[:, :4], original_shape).round()
            
            for *xyxy, conf, cls in reversed(pred):
                class_id = int(cls)
                class_name = self.names[class_id]
                confidence = float(conf)
                
                # 바운딩 박스 좌표
                x_min, y_min, x_max, y_max = [int(coord) for coord in xyxy]
                
                detection = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x_min, y_min, x_max, y_max]
                }
                
                detections.append(detection)
                
                logger.info(f"탐지: {class_name} ({confidence:.2f}) at [{x_min}, {y_min}, {x_max}, {y_max}]")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """탐지 결과를 프레임에 그리기"""
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            x_min, y_min, x_max, y_max = detection["bbox"]
            
            # 바운딩 박스 그리기
            color = (0, 255, 0)  # 초록색
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # 라벨 그리기
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 라벨 배경
            cv2.rectangle(frame, (x_min, y_min - label_size[1] - 10), 
                         (x_min + label_size[0], y_min), color, -1)
            
            # 라벨 텍스트
            cv2.putText(frame, label, (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "initialized": self.is_initialized,
            "device": str(self.device) if self.device else None,
            "classes": self.names if self.names else None,
            "conf_threshold": self.conf_thres,
            "iou_threshold": self.iou_thres,
            "available": YOLO_AVAILABLE
        }