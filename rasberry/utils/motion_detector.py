"""
움직임 감지 모듈
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List

class MotionDetector:
    def __init__(self, threshold: int = 5000, blur_size: int = 21, min_area: int = 1000):
        self.threshold = threshold
        self.blur_size = blur_size
        self.min_area = min_area
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        self.last_motion_time = 0
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """
        프레임에서 움직임 감지
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (motion_detected, bounding_boxes)
        """
        # 그레이스케일 변환 (이미 그레이스케일인 경우 스킵)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]  # 단일 채널 추출
        else:
            gray = frame  # 이미 그레이스케일
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # 배경 차분 적용
        fg_mask = self.background_subtractor.apply(blurred)
        
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        bounding_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                motion_detected = True
        
        if motion_detected:
            self.last_motion_time = time.time()
            
        return motion_detected, bounding_boxes
    
    def is_motion_recent(self, seconds: int = 5) -> bool:
        """최근 N초 내에 움직임이 있었는지 확인"""
        return (time.time() - self.last_motion_time) < seconds
    
    def draw_motion_areas(self, frame: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """움직임 영역을 프레임에 표시"""
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame