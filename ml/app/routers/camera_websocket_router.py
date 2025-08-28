"""
카메라 웹소켓 라우터
라즈베리파이 카메라들과의 실시간 통신 처리
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import asyncio
import json
import logging
import base64
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional

from app.services.yolo_service import YOLOService
from app.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)

router = APIRouter()

class CameraConnectionManager:
    """카메라 연결 관리자"""
    
    def __init__(self):
        self.active_cameras: Dict[str, Dict] = {}
        self.yolo_service = YOLOService()
        self.metadata_service = MetadataService()
        
        # 녹화 상태 관리
        self.recording_sessions: Dict[str, Dict] = {}
        
        # 프레임 버퍼 (HQ 프레임 임시 저장)
        self.frame_buffers: Dict[str, List] = {}
        
    async def connect(self, websocket: WebSocket, camera_id: str):
        """카메라 연결"""
        await websocket.accept()
        
        self.active_cameras[camera_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "last_frame_time": 0,
            "frame_count": 0,
            "gh_idx": None,
            "config": {}
        }
        
        self.frame_buffers[camera_id] = []
        
        logger.info(f"카메라 연결됨: {camera_id}")
    
    def disconnect(self, camera_id: str):
        """카메라 연결 해제"""
        if camera_id in self.active_cameras:
            del self.active_cameras[camera_id]
        
        if camera_id in self.recording_sessions:
            del self.recording_sessions[camera_id]
            
        if camera_id in self.frame_buffers:
            del self.frame_buffers[camera_id]
            
        logger.info(f"카메라 연결 해제됨: {camera_id}")
    
    async def handle_camera_init(self, camera_id: str, message: dict):
        """카메라 초기화 메시지 처리"""
        if camera_id not in self.active_cameras:
            return
            
        camera_info = self.active_cameras[camera_id]
        camera_info.update({
            "gh_idx": message.get("gh_idx"),
            "config": message.get("config", {})
        })
        
        logger.info(f"카메라 초기화 완료: {camera_id} (GH_IDX: {message.get('gh_idx')})")
    
    async def handle_frame_data(self, camera_id: str, message: dict):
        """프레임 데이터 처리"""
        if camera_id not in self.active_cameras:
            return
        
        camera_info = self.active_cameras[camera_id]
        camera_info["last_frame_time"] = time.time()
        camera_info["frame_count"] += 1
        
        frame_type = message.get("frame_type")
        motion_detected = message.get("motion_detected", False)
        
        # Base64 디코딩하여 프레임 복원
        frame = self._decode_frame(message.get("frame_data"))
        if frame is None:
            logger.error(f"프레임 디코딩 실패: {camera_id}")
            return
        
        # HQ 프레임이면 YOLO 탐지 수행
        if frame_type == "hq" and motion_detected:
            await self._process_detection(camera_id, frame, message)
        
        # 통계 로깅
        if camera_info["frame_count"] % 100 == 0:  # 100 프레임마다
            fps = camera_info["frame_count"] / (time.time() - camera_info["connected_at"])
            logger.info(f"카메라 {camera_id} 통계: FPS {fps:.1f}, 총 프레임 {camera_info['frame_count']}")
    
    async def handle_recording_event(self, camera_id: str, message: dict):
        """녹화 이벤트 처리"""
        event_type = message.get("event_type")
        gh_idx = message.get("gh_idx")
        
        if event_type == "recording_start":
            # 녹화 세션 시작
            session_id = f"{camera_id}_{int(time.time())}"
            self.recording_sessions[camera_id] = {
                "session_id": session_id,
                "started_at": time.time(),
                "gh_idx": gh_idx,
                "frame_count": 0,
                "detected_insects": []
            }
            
            logger.info(f"녹화 시작: {camera_id} -> {session_id}")
            
        elif event_type == "recording_stop":
            # 녹화 세션 종료
            if camera_id in self.recording_sessions:
                session = self.recording_sessions[camera_id]
                duration = time.time() - session["started_at"]
                
                logger.info(f"녹화 종료: {camera_id} -> {session['session_id']} "
                          f"({duration:.1f}초, {session['frame_count']}프레임)")
                
                # 세션 정리
                del self.recording_sessions[camera_id]
        
        elif event_type == "motion_detected":
            motion_areas = message.get("motion_areas", [])
            logger.info(f"움직임 감지: {camera_id} -> {len(motion_areas)}개 영역")
    
    def _decode_frame(self, base64_data: str) -> Optional[np.ndarray]:
        """Base64 데이터를 OpenCV 프레임으로 변환"""
        try:
            # Base64 디코딩
            img_data = base64.b64decode(base64_data)
            
            # bytes를 numpy array로 변환
            nparr = np.frombuffer(img_data, np.uint8)
            
            # JPEG 디코딩
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            logger.error(f"프레임 디코딩 오류: {e}")
            return None
    
    async def _process_detection(self, camera_id: str, frame: np.ndarray, message: dict):
        """YOLO 탐지 수행"""
        try:
            # YOLO 탐지 실행
            detections = await self.yolo_service.detect_insects(frame)
            
            if detections:
                gh_idx = message.get("gh_idx")
                timestamp = message.get("timestamp")
                
                logger.info(f"해충 탐지! 카메라: {camera_id}, 탐지 수: {len(detections)}")
                
                # 각 탐지 결과 처리
                for detection in detections:
                    await self._handle_detection_result(
                        camera_id, detection, frame, gh_idx, timestamp
                    )
        
        except Exception as e:
            logger.error(f"탐지 처리 오류 {camera_id}: {e}")
    
    async def _handle_detection_result(self, camera_id: str, detection: dict, 
                                     frame: np.ndarray, gh_idx: int, timestamp: float):
        """개별 탐지 결과 처리"""
        try:
            insect_name = detection["class_name"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            
            # 메타데이터 저장
            metadata = await self.metadata_service.save_detection_metadata(
                camera_id=camera_id,
                gh_idx=gh_idx,
                insect_name=insect_name,
                confidence=confidence,
                bbox=bbox,
                timestamp=timestamp
            )
            
            # 탐지된 영역 크롭하여 저장
            cropped_frame = self._crop_detection(frame, bbox)
            if cropped_frame is not None:
                img_path = await self._save_detection_image(
                    camera_id, cropped_frame, metadata["rec_id"], metadata["track_id"]
                )
                
                # Spring Boot API로 데이터 전송
                await self._send_to_spring_boot(
                    insect_name, confidence, img_path, gh_idx
                )
            
            # 녹화 세션에 탐지 정보 추가
            if camera_id in self.recording_sessions:
                session = self.recording_sessions[camera_id]
                session["detected_insects"].append({
                    "insect_name": insect_name,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "bbox": bbox
                })
        
        except Exception as e:
            logger.error(f"탐지 결과 처리 오류: {e}")
    
    def _crop_detection(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """탐지 영역 크롭"""
        try:
            x_min, y_min, x_max, y_max = bbox
            cropped = frame[y_min:y_max, x_min:x_max]
            return cropped if cropped.size > 0 else None
        except Exception as e:
            logger.error(f"크롭 오류: {e}")
            return None
    
    async def _save_detection_image(self, camera_id: str, cropped_frame: np.ndarray, 
                                  rec_id: int, track_id: int) -> str:
        """탐지 이미지 저장"""
        # 저장 디렉토리 생성
        save_dir = Path("data/detections") / camera_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"rec{rec_id}_track{track_id}_{timestamp}.jpg"
        filepath = save_dir / filename
        
        # 이미지 저장
        cv2.imwrite(str(filepath), cropped_frame)
        
        return str(filepath)
    
    async def _send_to_spring_boot(self, insect_name: str, confidence: float, 
                                 img_path: str, gh_idx: int):
        """Spring Boot API로 탐지 결과 전송"""
        # 기존 detect.py의 send_detection_to_api 로직과 동일
        import requests
        from datetime import datetime
        
        def get_insect_idx(name):
            return {
                "꽃노랑총채벌레": 1,
                "담배가루이": 2,
                "비단노린재": 3,
                "알락수염노린재": 4
            }.get(name, 0)
        
        now = datetime.now()
        created_at = now.strftime("%Y-%m-%d %H:%M:%S")
        
        payload = {
            "anlsModel": "YOLOv5",
            "anlsContent": f"{insect_name} {confidence * 100:.2f}%로 탐지완료",
            "anlsResult": insect_name,
            "createdAt": created_at,
            "insectIdx": get_insect_idx(insect_name),
            "imgIdx": 1,  # 임시값 - 실제로는 이미지 업로드 후 받은 값 사용
            "notiCheck": 'N',
            "ghIdx": gh_idx,
            "anlsAcc": int(confidence * 100)
        }
        
        try:
            res = requests.post("http://localhost:8095/api/qc-classification", json=payload)
            logger.info(f"Spring Boot 전송 완료: {insect_name} | 상태: {res.status_code}")
            
            # 전화 발신
            await self._make_call(gh_idx, insect_name, confidence)
            
        except Exception as e:
            logger.error(f"Spring Boot 전송 실패: {e}")
    
    async def _make_call(self, gh_idx: int, insect_name: str, confidence: float):
        """전화 발신"""
        import requests
        
        try:
            ml_api_url = "http://localhost:8003/api/make-call"
            params = {
                "gh_idx": gh_idx,
                "insect_name": insect_name,
                "confidence": confidence
            }
            
            response = requests.post(ml_api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"전화 발신 성공: {insect_name} (신뢰도: {confidence:.2f})")
            else:
                logger.error(f"전화 발신 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"전화 발신 오류: {e}")
    
    def get_camera_stats(self) -> dict:
        """카메라 통계 반환"""
        stats = {}
        current_time = time.time()
        
        for camera_id, info in self.active_cameras.items():
            uptime = current_time - info["connected_at"]
            fps = info["frame_count"] / uptime if uptime > 0 else 0
            
            stats[camera_id] = {
                "connected": True,
                "uptime": uptime,
                "frame_count": info["frame_count"],
                "fps": fps,
                "last_frame_ago": current_time - info["last_frame_time"],
                "gh_idx": info.get("gh_idx"),
                "config": info.get("config", {})
            }
        
        return stats

# 전역 연결 관리자
camera_manager = CameraConnectionManager()

@router.websocket("/ws/camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    """카메라 웹소켓 엔드포인트"""
    camera_id = None
    
    try:
        await websocket.accept()
        
        # 연결 메시지 대기
        data = await websocket.receive_text()
        message = json.loads(data)
        
        if message.get("type") == "camera_init":
            camera_id = message.get("camera_id")
            if not camera_id:
                await websocket.send_text(json.dumps({"error": "camera_id required"}))
                return
            
            await camera_manager.connect(websocket, camera_id)
            await camera_manager.handle_camera_init(camera_id, message)
            
            # 연결 확인 메시지 전송
            await websocket.send_text(json.dumps({
                "type": "connection_confirmed",
                "camera_id": camera_id,
                "server_time": time.time()
            }))
        
        # 메시지 루프
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "frame_data":
                    await camera_manager.handle_frame_data(camera_id, message)
                    
                elif message_type == "recording_event":
                    await camera_manager.handle_recording_event(camera_id, message)
                
                elif message_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"메시지 처리 오류 {camera_id}: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"웹소켓 오류: {e}")
    finally:
        if camera_id:
            camera_manager.disconnect(camera_id)

@router.get("/api/camera/stats")
async def get_camera_stats():
    """카메라 통계 조회"""
    return {
        "cameras": camera_manager.get_camera_stats(),
        "total_cameras": len(camera_manager.active_cameras)
    }