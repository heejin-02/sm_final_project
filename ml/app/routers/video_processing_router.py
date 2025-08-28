"""
비디오 버퍼 처리 라우터
라즈베리파이에서 받은 10초간의 LQ+HQ 프레임을 처리하여 해충 탐지 및 분류
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import base64
import numpy as np
import cv2
from datetime import datetime
from typing import List, Optional
import asyncio

from app.services.yolo_service import YOLOService
from app.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)

router = APIRouter()

class VideoBufferRequest(BaseModel):
    type: str = "video_buffer"
    camera_id: str
    gh_idx: int
    recording_start_time: float
    recording_duration: int
    frame_count: int
    lq_frames: List[str]  # Base64 인코딩된 LQ 프레임들
    hq_frames: List[str]  # Base64 인코딩된 HQ 프레임들
    timestamps: List[float]
    lq_resolution: List[int]
    hq_resolution: List[int]

class VideoProcessingResponse(BaseModel):
    success: bool
    message: str
    camera_id: str
    total_frames: int
    processing_time: float
    detections: Optional[List[dict]] = None

# 서비스 인스턴스들
yolo_service = YOLOService()
metadata_service = MetadataService()

@router.post("/process-video-buffer", response_model=VideoProcessingResponse)
async def process_video_buffer(request: VideoBufferRequest):
    """
    10초간의 비디오 버퍼를 처리하여 해충 탐지 및 크롭 이미지 생성
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🎬 비디오 버퍼 처리 시작: camera_id={request.camera_id}, frames={request.frame_count}")
        
        # 프레임 수 검증
        if len(request.lq_frames) != len(request.hq_frames):
            raise HTTPException(status_code=400, detail="LQ와 HQ 프레임 수가 일치하지 않습니다")
        
        if len(request.lq_frames) != request.frame_count:
            raise HTTPException(status_code=400, detail="프레임 수가 일치하지 않습니다")
        
        # 모든 탐지 결과 저장
        all_detections = []
        processed_frames = 0
        
        # 프레임별 처리
        for i, (lq_b64, hq_b64, timestamp) in enumerate(zip(request.lq_frames, request.hq_frames, request.timestamps)):
            try:
                # LQ 프레임 디코딩
                lq_frame = decode_base64_frame(lq_b64)
                if lq_frame is None:
                    logger.warning(f"LQ 프레임 {i} 디코딩 실패")
                    continue
                
                # LQ 프레임에서 YOLO 탐지 수행
                detections = await yolo_service.detect_insects(lq_frame)
                
                if detections:
                    # HQ 프레임 디코딩
                    hq_frame = decode_base64_frame(hq_b64)
                    if hq_frame is None:
                        logger.warning(f"HQ 프레임 {i} 디코딩 실패")
                        continue
                    
                    # 탐지 결과를 HQ 프레임과 동기화하여 처리
                    frame_detections = await process_detections_with_hq_sync(
                        detections, lq_frame, hq_frame, request, timestamp, i
                    )
                    
                    all_detections.extend(frame_detections)
                    
                    logger.info(f"프레임 {i}/{request.frame_count}: {len(detections)}개 탐지")
                
                processed_frames += 1
                
                # 진행상황 로깅 (20프레임마다)
                if i % 20 == 0:
                    logger.info(f"처리 진행: {i+1}/{request.frame_count} ({((i+1)/request.frame_count)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"프레임 {i} 처리 중 오류: {e}")
                continue
        
        # 처리 완료
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ 비디오 처리 완료: {processed_frames}/{request.frame_count}프레임, "
                   f"{len(all_detections)}개 탐지, {processing_time:.2f}초 소요")
        
        return VideoProcessingResponse(
            success=True,
            message=f"{len(all_detections)}개 해충 탐지 완료",
            camera_id=request.camera_id,
            total_frames=processed_frames,
            processing_time=processing_time,
            detections=all_detections
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ 비디오 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

async def process_detections_with_hq_sync(detections, lq_frame, hq_frame, request, timestamp, frame_idx):
    """
    LQ에서 탐지된 결과를 HQ 프레임과 동기화하여 크롭 이미지 생성
    """
    frame_detections = []
    
    # 해상도 비율 계산 (LQ → HQ 스케일링)
    lq_h, lq_w = lq_frame.shape[:2]
    hq_h, hq_w = hq_frame.shape[:2]
    
    scale_x = hq_w / lq_w
    scale_y = hq_h / lq_h
    
    for detection in detections:
        try:
            # LQ bbox를 HQ 좌표계로 변환
            lq_bbox = detection["bbox"]  # [x_min, y_min, x_max, y_max]
            
            hq_bbox = [
                int(lq_bbox[0] * scale_x),  # x_min
                int(lq_bbox[1] * scale_y),  # y_min  
                int(lq_bbox[2] * scale_x),  # x_max
                int(lq_bbox[3] * scale_y)   # y_max
            ]
            
            # 경계 검사
            hq_bbox[0] = max(0, min(hq_bbox[0], hq_w-1))
            hq_bbox[1] = max(0, min(hq_bbox[1], hq_h-1))
            hq_bbox[2] = max(hq_bbox[0]+1, min(hq_bbox[2], hq_w))
            hq_bbox[3] = max(hq_bbox[1]+1, min(hq_bbox[3], hq_h))
            
            # HQ 프레임에서 크롭
            cropped_hq = hq_frame[hq_bbox[1]:hq_bbox[3], hq_bbox[0]:hq_bbox[2]]
            
            if cropped_hq.size == 0:
                logger.warning(f"크롭 영역이 비어있음: {hq_bbox}")
                continue
            
            # 메타데이터 저장
            metadata = await metadata_service.save_detection_metadata(
                camera_id=request.camera_id,
                gh_idx=request.gh_idx,
                insect_name=detection["class_name"],
                confidence=detection["confidence"],
                bbox=hq_bbox,
                timestamp=timestamp
            )
            
            # 크롭 이미지 저장
            crop_path = await save_cropped_image(
                request.camera_id, cropped_hq, metadata["rec_id"], 
                metadata["track_id"], frame_idx, detection["class_name"]
            )
            
            # Spring Boot로 결과 전송
            await send_detection_to_spring_boot(
                detection["class_name"], detection["confidence"], crop_path, request.gh_idx
            )
            
            frame_detections.append({
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "lq_bbox": lq_bbox,
                "hq_bbox": hq_bbox,
                "rec_id": metadata["rec_id"],
                "track_id": metadata["track_id"],
                "crop_path": crop_path,
                "frame_index": frame_idx,
                "timestamp": timestamp
            })
            
        except Exception as e:
            logger.error(f"탐지 결과 처리 중 오류: {e}")
            continue
    
    return frame_detections

def decode_base64_frame(base64_data: str) -> Optional[np.ndarray]:
    """Base64 데이터를 OpenCV 프레임으로 변환"""
    try:
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"프레임 디코딩 오류: {e}")
        return None

async def save_cropped_image(camera_id: str, cropped_frame: np.ndarray, 
                            rec_id: int, track_id: int, frame_idx: int, class_name: str) -> str:
    """크롭된 고해상도 이미지 저장"""
    from pathlib import Path
    
    # 저장 디렉토리 생성 (곤충 종류별로 분류)
    save_dir = Path("data/cropped_detections") / camera_id / class_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"rec{rec_id}_track{track_id}_frame{frame_idx}_{timestamp}.jpg"
    filepath = save_dir / filename
    
    # 이미지 저장
    cv2.imwrite(str(filepath), cropped_frame)
    
    return str(filepath)

async def send_detection_to_spring_boot(insect_name: str, confidence: float, 
                                      crop_path: str, gh_idx: int):
    """Spring Boot API로 탐지 결과 전송"""
    import requests
    
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
        "imgIdx": 1,
        "notiCheck": 'N',
        "ghIdx": gh_idx,
        "anlsAcc": int(confidence * 100)
    }
    
    try:
        res = requests.post("http://localhost:8095/api/qc-classification", json=payload)
        logger.info(f"Spring Boot 전송 완료: {insect_name} | 상태: {res.status_code}")
        
        # 전화 발신
        await make_call(gh_idx, insect_name, confidence)
        
    except Exception as e:
        logger.error(f"Spring Boot 전송 실패: {e}")

async def make_call(gh_idx: int, insect_name: str, confidence: float):
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