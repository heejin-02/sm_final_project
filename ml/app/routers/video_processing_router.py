"""
비디오 버퍼 처리 라우터 (대표 1건만 Spring Boot 전송 + 전화 발신 + 단일 스트림 허용)
라즈베리파이에서 받은 10초간의 프레임(LQ 단일 스트림 또는 LQ+HQ)을 처리하여 해충 탐지 및 분류
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import base64
import numpy as np
import cv2
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import os
from PIL import ImageFont, ImageDraw, Image

from app.services.yolo_service import YOLOService
from app.services.metadata_service import MetadataService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ----------------------------
# 요청/응답 모델
# ----------------------------
class VideoBufferRequest(BaseModel):
    type: str = "video_buffer"
    camera_id: str
    gh_idx: int
    recording_start_time: float
    recording_duration: int
    frame_count: int
    lq_frames: List[str]                      # Base64 LQ 프레임들 (필수)
    hq_frames: Optional[List[str]] = None     # Base64 HQ 프레임들 (옵션: 없으면 LQ로 대체)
    timestamps: List[float]
    lq_resolution: List[int]                  # [width, height]
    hq_resolution: Optional[List[int]] = None # [width, height] (옵션: 없으면 LQ로 대체)

class VideoProcessingResponse(BaseModel):
    success: bool
    message: str
    camera_id: str
    total_frames: int
    processing_time: float
    detections: Optional[List[dict]] = None

# ----------------------------
# 서비스 인스턴스
# ----------------------------
yolo_service = YOLOService()
metadata_service = MetadataService()

# ----------------------------
# 엔드포인트
# ----------------------------
@router.post("/process-video-buffer", response_model=VideoProcessingResponse)
async def process_video_buffer(request: VideoBufferRequest):
    """
    10초간의 비디오 버퍼를 처리하여 해충 탐지 및 크롭 생성/저장, 시각화 비디오 업로드,
    대표 탐지 1건 전송 및 전화 발신.
    """
    start_time = datetime.now()

    try:
        logger.info(f"🎬 비디오 버퍼 처리 시작: camera_id={request.camera_id}, frames={request.frame_count}")

        # ----------------------------
        # 입력 정규화
        # ----------------------------
        lq_frames = request.lq_frames or []
        hq_frames = request.hq_frames or []
        timestamps = request.timestamps or []

        if len(lq_frames) == 0:
            raise HTTPException(status_code=400, detail="LQ 프레임이 비어 있습니다")

        # 단일 스트림 허용: HQ 없으면 LQ 재사용
        if not hq_frames:
            logger.info("단일 스트림 모드 감지: HQ 프레임이 없어 LQ를 HQ로 재사용합니다.")
            hq_frames = lq_frames
            if not request.hq_resolution:
                request.hq_resolution = request.lq_resolution

        # 기준 프레임 수 보정
        n = len(lq_frames)
        if request.frame_count != n:
            logger.warning(f"frame_count({request.frame_count}) ≠ 실제 LQ 길이({n}) → 보정")
        request.frame_count = n

        # 길이 불일치 시 가장 짧은 길이로 절단
        if len(hq_frames) != n or len(timestamps) != n:
            m = min(n, len(hq_frames), len(timestamps))
            if m == 0:
                raise HTTPException(status_code=400, detail="프레임/타임스탬프 길이가 0입니다")
            if len(hq_frames) != n:
                logger.warning(f"HQ 길이({len(hq_frames)}) ≠ LQ 길이({n}) → {m}으로 보정")
            if len(timestamps) != n:
                logger.warning(f"timestamps 길이({len(timestamps)}) ≠ LQ 길이({n}) → {m}으로 보정")
            lq_frames, hq_frames, timestamps = lq_frames[:m], hq_frames[:m], timestamps[:m]
            request.frame_count = m

        # ----------------------------
        # 프레임 처리 루프
        # ----------------------------
        all_detections: List[dict] = []
        annotated_hq_frames: List[Optional[np.ndarray]] = []
        processed_frames = 0

        for i in range(request.frame_count):
            lq_b64 = lq_frames[i]
            hq_b64 = hq_frames[i]
            ts = timestamps[i]

            try:
                # 디코딩
                hq_frame = decode_base64_frame(hq_b64)
                if hq_frame is None:
                    logger.warning(f"HQ 프레임 {i} 디코딩 실패")
                    annotated_hq_frames.append(None)
                    continue

                lq_frame = decode_base64_frame(lq_b64)
                if lq_frame is None:
                    logger.warning(f"LQ 프레임 {i} 디코딩 실패")
                    annotated_hq_frames.append(hq_frame.copy())
                    continue

                # YOLO 탐지 (LQ 기준)
                detections = await yolo_service.detect_insects(lq_frame)

                # HQ 시각화 프레임 생성
                annotated_hq = draw_detections_on_hq_frame(
                    detections, lq_frame, hq_frame
                ) if detections else hq_frame.copy()
                annotated_hq_frames.append(annotated_hq)

                # 메타/크롭 저장
                if detections:
                    frame_detections = await process_detections_with_hq_sync(
                        detections, lq_frame, hq_frame, request, ts, i
                    )
                    all_detections.extend(frame_detections)

                processed_frames += 1

                if i % 20 == 0:
                    logger.info(f"처리 진행: {i+1}/{request.frame_count} ({((i+1)/request.frame_count)*100:.1f}%)")

            except Exception as e:
                logger.error(f"프레임 {i} 처리 중 오류: {e}")
                continue

        # ----------------------------
        # 시각화 비디오 생성 → Spring Boot 전송
        # ----------------------------
        video_path, img_idx = None, None
        if annotated_hq_frames:
            video_path = await create_annotated_video(annotated_hq_frames, request)
            if video_path:
                img_idx = await send_video_to_spring_boot(video_path, request, len(all_detections))
                # 대표 1건만 전송 + 전화 발신
                if img_idx and all_detections:
                    await send_all_detections_to_spring_boot(all_detections, request.gh_idx, img_idx)

        # ----------------------------
        # 응답
        # ----------------------------
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 비디오 처리 완료: {processed_frames}/{request.frame_count}프레임, "
            f"{len(all_detections)}개 탐지, {processing_time:.2f}초 소요"
        )

        return VideoProcessingResponse(
            success=True,
            message=f"{len(all_detections)}개 해충 탐지 완료, 비디오 생성: {video_path}",
            camera_id=request.camera_id,
            total_frames=processed_frames,
            processing_time=processing_time,
            detections=all_detections
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 비디오 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

# ----------------------------
# 헬퍼 함수들
# ----------------------------
async def process_detections_with_hq_sync(
    detections, lq_frame, hq_frame, request: VideoBufferRequest, timestamp, frame_idx
):
    """
    LQ에서 탐지된 결과를 HQ 좌표계로 변환하여 크롭 이미지 저장 및 메타데이터 기록
    """
    frame_detections: List[dict] = []

    lq_h, lq_w = lq_frame.shape[:2]
    hq_h, hq_w = hq_frame.shape[:2]
    scale_x, scale_y = hq_w / max(1, lq_w), hq_h / max(1, lq_h)

    for det in detections:
        try:
            lq_bbox = det["bbox"]  # [x_min, y_min, x_max, y_max]
            hq_bbox = [
                int(lq_bbox[0] * scale_x),
                int(lq_bbox[1] * scale_y),
                int(lq_bbox[2] * scale_x),
                int(lq_bbox[3] * scale_y),
            ]
            # 경계 보정
            hq_bbox[0] = max(0, min(hq_bbox[0], hq_w - 1))
            hq_bbox[1] = max(0, min(hq_bbox[1], hq_h - 1))
            hq_bbox[2] = max(hq_bbox[0] + 1, min(hq_bbox[2], hq_w))
            hq_bbox[3] = max(hq_bbox[1] + 1, min(hq_bbox[3], hq_h))

            cropped_hq = hq_frame[hq_bbox[1]:hq_bbox[3], hq_bbox[0]:hq_bbox[2]]
            if cropped_hq.size == 0:
                logger.warning(f"크롭 영역이 비어있음: {hq_bbox}")
                continue

            # 메타 저장
            metadata = await metadata_service.save_detection_metadata(
                camera_id=request.camera_id,
                gh_idx=request.gh_idx,
                insect_name=det["class_name"],
                confidence=det["confidence"],
                bbox=hq_bbox,
                timestamp=timestamp
            )

            # 크롭 저장
            crop_path = await save_cropped_image(
                request.camera_id, cropped_hq, metadata["rec_id"],
                metadata["track_id"], frame_idx, det["class_name"]
            )

            frame_detections.append({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
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
    """Base64 → OpenCV 이미지"""
    try:
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"프레임 디코딩 오류: {e}")
        return None


async def save_cropped_image(
    camera_id: str, cropped_frame: np.ndarray, rec_id: int, track_id: int, frame_idx: int, class_name: str
) -> str:
    """크롭된 HQ 이미지 저장"""
    save_dir = Path("data/cropped_detections") / camera_id / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"rec{rec_id}_track{track_id}_frame{frame_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    filepath = save_dir / filename
    cv2.imwrite(str(filepath), cropped_frame)
    return str(filepath)


async def send_detection_to_spring_boot(
    insect_name: str, confidence: float, crop_path: str, gh_idx: int, img_idx: int = None
):
    """대표 탐지 1건 전송"""
    import requests

    payload = {
        "anlsModel": "YOLOv5",
        "anlsContent": f"{insect_name} {confidence * 100:.2f}%로 탐지완료",
        "anlsResult": insect_name,
        "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "insectIdx": {"꽃노랑총채벌레": 1, "담배가루이": 2, "비단노린재": 3, "알락수염노린재": 4}.get(insect_name, 0),
        "imgIdx": img_idx if img_idx else 1,
        "notiCheck": 'N',
        "ghIdx": gh_idx,
        "anlsAcc": int(confidence * 100),
    }

    try:
        spring_boot_url = os.getenv("SPRING_BOOT_URL", "http://localhost:8095")
        res = requests.post(f"{spring_boot_url}/api/qc-classification", json=payload, timeout=10)
        logger.info(f"Spring Boot 전송 완료: {insect_name} | 상태: {res.status_code}")
    except Exception as e:
        logger.error(f"Spring Boot 전송 실패: {e}")


def draw_text_korean(img, text, org, font_size=20, color=(0, 0, 0)):
    """이미지에 한글 텍스트 출력 (PIL 사용, 폰트 경로 폴백 지원)"""
    try:
        # 환경변수 우선 → 리눅스 나눔고딕 → 맥 → 윈도우 → 실패 시 OpenCV 폴백
        candidates = [
            os.getenv("KOREAN_FONT_PATH"),
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/Library/Fonts/AppleGothic.ttf",
            "C:/Windows/Fonts/NanumGothic.ttf",
            "C:/Windows/Fonts/malgun.ttf",
        ]
        font_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if not font_path:
            raise FileNotFoundError("한글 폰트를 찾을 수 없습니다")

        font = ImageFont.truetype(font_path, font_size)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(org, text, font=font, fill=color)
        return np.array(img_pil)

    except Exception as e:
        logger.warning(f"한글 텍스트 출력 오류(기본 폰트로 대체): {e}")
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img


def draw_detections_on_hq_frame(
    detections: List[dict],
    lq_frame: np.ndarray,
    hq_frame: np.ndarray,
    convert_to_rgb: bool = False
) -> np.ndarray:
    """
    HQ 프레임에 LQ에서 탐지된 바운딩박스를 시각화
    """
    annotated = hq_frame.copy()

    # 해상도 비율(LQ → HQ)
    lq_h, lq_w = lq_frame.shape[:2]
    hq_h, hq_w = hq_frame.shape[:2]
    scale_x, scale_y = hq_w / max(1, lq_w), hq_h / max(1, lq_h)

    # 곤충별 색상
    colors = {
        "꽃노랑총채벌레": (0, 255, 255),  # 노랑
        "담배가루이": (255, 255, 255),    # 하양
        "비단노린재": (0, 255, 0),        # 초록
        "알락수염노린재": (255, 0, 0),     # 빨강
    }

    for det in detections:
        bbox = det["bbox"]  # [x_min, y_min, x_max, y_max]
        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        # 경계 보정
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(hq_w - 1, x2), min(hq_h - 1, y2)

        class_name = det["class_name"]
        conf = det["confidence"]
        color = colors.get(class_name, (0, 255, 0))

        # 박스
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 레이블 배경
        label = f"{class_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)

        # 한글 텍스트
        annotated = draw_text_korean(annotated, label, (x1 + 5, y1 - 25), font_size=20, color=(0, 0, 0))

    if convert_to_rgb:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated


async def create_annotated_video(
    annotated_frames: List[Optional[np.ndarray]], request: VideoBufferRequest
) -> Optional[str]:
    """
    시각화된 프레임들로 MP4(H.264 우선) 비디오 생성
    """
    try:
        video_dir = Path("data/processed_videos") / request.camera_id
        video_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        video_path = video_dir / f"detection_{request.camera_id}_{ts}.mp4"

        # 유효 프레임 찾기
        valid = next((f for f in annotated_frames if f is not None), None)
        if valid is None:
            logger.error("유효한 프레임이 없습니다")
            return None

        height, width = valid.shape[:2]
        # FPS 추정: frame_count / duration (1~30 사이 clamp)
        est_fps = int(round(max(1.0, min(30.0, request.frame_count / max(1e-3, request.recording_duration)))))
        fps = est_fps if est_fps > 0 else 10

        # imageio-ffmpeg 우선(H.264)
        try:
            import imageio
            logger.info("🎬 imageio-ffmpeg를 사용하여 H.264 비디오 생성 시작")

            rgb_frames = []
            for i, frame in enumerate(annotated_frames):
                if frame is None:
                    # 빈 프레임 → 직전 프레임 또는 valid
                    frame = annotated_frames[i-1] if i > 0 and annotated_frames[i-1] is not None else valid
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # macro_block_size=16 (기본) → 일부 해상도는 내부에서 패딩될 수 있음(경고 로그 정상)
            imageio.mimwrite(
                str(video_path),
                rgb_frames,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                macro_block_size=16,
            )
            logger.info(f"✅ imageio-ffmpeg로 H.264 비디오 생성 완료: {video_path}")
            return str(video_path)

        except ImportError:
            logger.warning("⚠️ imageio-ffmpeg 미설치. OpenCV mp4v 코덱으로 폴백합니다.")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error("VideoWriter 열기 실패")
                return None

            for i, frame in enumerate(annotated_frames):
                if frame is None:
                    frame = annotated_frames[i-1] if i > 0 and annotated_frames[i-1] is not None else valid
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            out.release()

            logger.info(f"✅ 비디오 생성 완료: {video_path} (mp4v)")
            logger.warning("⚠️ mp4v 코덱은 일부 브라우저에서 재생 호환성이 낮을 수 있습니다.")
            return str(video_path)

    except Exception as e:
        logger.error(f"❌ 비디오 생성 실패: {e}")
        return None


async def send_video_to_spring_boot(
    video_path: str, request: VideoBufferRequest, detection_count: int
) -> Optional[int]:
    """
    생성된 비디오를 Spring Boot 서버로 업로드하고 IMG_IDX 반환
    """
    import requests

    try:
        if not Path(video_path).exists():
            logger.error(f"비디오 파일이 존재하지 않습니다: {video_path}")
            return None

        base_url = os.getenv("SPRING_BOOT_URL", "http://localhost:8095")
        spring_boot_url = f"{base_url}/api/video/upload"

        with open(video_path, 'rb') as video_file:
            files = {'video': (Path(video_path).name, video_file, 'video/mp4')}
            data = {
                'camera_id': request.camera_id,
                'gh_idx': request.gh_idx,
                'detection_count': detection_count,
                'recording_start_time': request.recording_start_time,
                'frame_count': request.frame_count
            }
            res = requests.post(spring_boot_url, files=files, data=data, timeout=30)

        if res.status_code == 200:
            result = res.json()
            img_idx = result.get('img_idx')
            logger.info(f"✅ Spring Boot 비디오 전송 성공: {video_path}, IMG_IDX: {img_idx}")
            return img_idx
        else:
            logger.error(f"❌ Spring Boot 비디오 전송 실패: {res.status_code} | {res.text}")
            return None

    except Exception as e:
        logger.error(f"❌ Spring Boot 비디오 전송 오류: {e}")
        return None


async def send_all_detections_to_spring_boot(detections: List[dict], gh_idx: int, img_idx: int):
    """
    누적된 탐지들 중 '대표 1건(최고 신뢰도)'만 전송하고 전화 발신
    """
    try:
        if not detections:
            logger.info("탐지 결과 없음 → 전송 생략")
            return
        best = max(detections, key=lambda x: x["confidence"])
        await send_detection_to_spring_boot(
            best["class_name"], best["confidence"], best["crop_path"], gh_idx, img_idx
        )
        logger.info(f"대표 탐지 전송 완료: {best['class_name']} ({best['confidence']:.2f})")

        # 전화 발신
        await make_call(gh_idx, best["class_name"], best["confidence"])

    except Exception as e:
        logger.error(f"❌ 대표 탐지 전송 실패: {e}")


async def make_call(gh_idx: int, insect_name: str, confidence: float):
    """전화 발신"""
    import requests
    try:
        ml_api_url = os.getenv("ML_API_URL", "http://localhost:8003/api/make-call")
        params = {
            "gh_idx": gh_idx,
            "insect_name": insect_name,
            "confidence": confidence
        }
        res = requests.post(ml_api_url, params=params, timeout=10)
        if res.status_code == 200:
            logger.info(f"전화 발신 성공: {insect_name} (신뢰도 {confidence:.2f})")
        else:
            logger.error(f"전화 발신 실패: {res.status_code} | {res.text}")
    except Exception as e:
        logger.error(f"전화 발신 오류: {e}")
