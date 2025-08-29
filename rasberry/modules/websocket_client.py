"""
웹소켓 클라이언트 모듈
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WebSocketClient:
    """웹소켓 클라이언트"""
    
    def __init__(self, server_host: str, server_port: int, websocket_endpoint: str, 
                 camera_id: str, gh_idx: int):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket_endpoint = websocket_endpoint
        self.camera_id = camera_id
        self.gh_idx = gh_idx
        
        self.websocket = None
        self.uri = f"ws://{server_host}:{server_port}{websocket_endpoint}"
    
    async def connect(self) -> bool:
        """웹소켓 연결"""
        try:
            logger.info(f"🔌 웹소켓 연결 시도: {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            logger.info("✅ 웹소켓 연결 성공")
            
            # 초기 연결 메시지 전송
            await self.send_init_message()
            return True
            
        except Exception as e:
            logger.error(f"❌ 웹소켓 연결 실패: {e}")
            self.websocket = None
            return False
    
    async def send_init_message(self, camera_type: str = "opencv"):
        """초기화 메시지 전송"""
        init_message = {
            "type": "camera_init",
            "camera_id": self.camera_id,
            "gh_idx": self.gh_idx,
            "config": {
                "camera_type": camera_type
            }
        }
        await self.send_message(init_message)
        logger.info("📤 초기화 메시지 전송 완료")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """메시지 전송"""
        if not self.websocket:
            return False
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"메시지 전송 오류: {e}")
            self.websocket = None
            return False
    
    async def send_frame_data(self, frame_data: str, frame_type: str, 
                            motion_detected: bool = False, motion_areas: list = None,
                            frame_size: int = 0, quality: int = 60, frame_shape: tuple = None):
        """프레임 데이터 전송"""
        message = {
            "type": "frame_data",
            "camera_id": self.camera_id,
            "gh_idx": self.gh_idx,
            "frame_type": frame_type,
            "timestamp": time.time(),
            "motion_detected": motion_detected,
            "motion_areas": motion_areas or [],
            "frame_data": frame_data,
            "frame_size": frame_size,
            "quality": quality,
            "frame_shape": frame_shape
        }
        return await self.send_message(message)
    
    async def send_recording_event(self, event_type: str, **kwargs):
        """녹화 이벤트 전송"""
        message = {
            "type": "recording_event",
            "camera_id": self.camera_id,
            "gh_idx": self.gh_idx,
            "event_type": event_type,
            "timestamp": time.time(),
            **kwargs
        }
        
        if await self.send_message(message):
            logger.info(f"녹화 이벤트 전송: {event_type}")
        else:
            logger.error(f"녹화 이벤트 전송 실패: {event_type}")
    
    async def send_video_buffer(self, lq_frames: list, hq_frames: list, 
                              timestamps: list, recording_start_time: float,
                              recording_duration: int, lq_resolution: tuple, 
                              hq_resolution: tuple):
        """비디오 버퍼 전송"""
        video_message = {
            "type": "video_buffer",
            "camera_id": self.camera_id,
            "gh_idx": self.gh_idx,
            "recording_start_time": recording_start_time,
            "recording_duration": recording_duration,
            "frame_count": len(lq_frames),
            "lq_frames": lq_frames,
            "hq_frames": hq_frames,
            "timestamps": timestamps,
            "lq_resolution": lq_resolution,
            "hq_resolution": hq_resolution
        }
        
        if await self.send_message(video_message):
            logger.info("✅ 비디오 버퍼 전송 완료")
        else:
            logger.error("❌ 비디오 버퍼 전송 실패")
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.websocket is not None
    
    async def close(self):
        """연결 종료"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            finally:
                self.websocket = None