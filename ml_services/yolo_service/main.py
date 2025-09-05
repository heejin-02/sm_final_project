"""
YOLO Inference Service
- 이미지/크롭/영상 입력 -> YOLO 추론 -> JSON 또는 시각화 결과 반환
- 메타데이터/DB 저장 없음 (stateless)
"""

import os, io, time, tempfile, shutil
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Windows에서 리눅스에서 저장한 체크포인트 열 때 PosixPath 문제 해결 ---
import pathlib
try:
    pathlib.PosixPath = pathlib.WindowsPath
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# ========================
# 설정
# ========================

# 기본 가중치 경로 (환경변수로 덮어쓸 수 있음)
_DEFAULT_WEIGHT = "C:/Users/smhrd1/Desktop/final_project11/sm_final_project/ml_services/yolo_service/model/last.pt"
DEFAULT_MODEL_PATH = os.getenv("YOLO_WEIGHTS", _DEFAULT_WEIGHT)
DEFAULT_MODEL_PATH = str(Path(DEFAULT_MODEL_PATH).resolve()).replace("\\", "/")

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_CONF = float(os.getenv("YOLO_CONF", 0.25))
DEFAULT_IOU  = float(os.getenv("YOLO_IOU", 0.45))
DEFAULT_IMGSZ = int(os.getenv("YOLO_IMGSZ", 640))


# ========================
# 유틸
# ========================

def to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def encode_jpeg(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ========================
# 메인 클래스
# ========================

class YoloInferenceService:
    """
    - 학습한 last.pt로 로드
    - 이미지/배치/영상 추론
    - 필요 시 시각화(BBox+라벨 한글 그대로) 반환
    """

    def __init__(
        self,
        weights: str = DEFAULT_MODEL_PATH,
        device: str = DEFAULT_DEVICE,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        force_reload: bool = False,
    ):
        self.device = device

        if not Path(weights).is_file():
            raise FileNotFoundError(f"[YOLO] weights not found: {weights}")

        print(f"[YOLO] Using weights: {weights}")
        print(f"[YOLO] Device: {self.device}")

        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=weights,
            force_reload=force_reload,
            trust_repo=True
        )

        self.model.to(self.device)
        if "cuda" in self.device:
            self.model.half()
        else:
            self.model.float()

        self.model.conf = conf
        self.model.iou = iou
        self.imgsz = imgsz
        self.names = self.model.names  # data.yaml의 한글 라벨 사용

    # -------- 이미지 1장 --------
    def infer_image(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        return_vis: bool = False,
    ) -> Dict[str, Any]:
        if conf is not None: self.model.conf = conf
        if iou is not None:  self.model.iou  = iou
        if imgsz is None:    imgsz = self.imgsz

        pil = to_pil(image_bgr)
        t0 = time.time()
        results = self.model(pil, size=imgsz)
        dt_ms = (time.time() - t0) * 1000

        dets = []
        h, w = image_bgr.shape[:2]
        for *xyxy, conf_score, cls_id in results.xyxy[0].tolist():
            x1, y1, x2, y2 = xyxy
            cid = int(cls_id)
            dets.append({
                "class_id": cid,
                "class_name": self.names[cid],
                "confidence": float(conf_score),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float((x1+x2)/2), float((y1+y2)/2), float(x2-x1), float(y2-y1)],
            })

        out = {
            "width": int(w),
            "height": int(h),
            "time_ms": round(dt_ms, 2),
            "detections": dets
        }

        if return_vis:
            vis_bgr = results.render()[0]
            out["visualization_jpeg"] = encode_jpeg(vis_bgr)

        return out

    # -------- 여러 크롭(배치) --------
    def infer_crops(
        self,
        crops_bgr: List[np.ndarray],
        **kwargs
    ) -> List[Dict[str, Any]]:
        return [self.infer_image(img, **kwargs) for img in crops_bgr]

    # -------- 동영상 파일 경로 -> 시각화된 동영상 파일 경로 --------
    def render_video_file(
        self,
        in_path: str,
        out_path: Optional[str] = None,
        stride: int = 1,
        out_fps: float = 0.0,
        **kwargs
    ) -> str:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {in_path}")

        in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = in_fps if out_fps <= 0 else out_fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # out_path 미지정 시: 입력 파일명 뒤에 _result_YYYYmmdd_HHMMSS.mp4
        if out_path is None or os.path.isdir(out_path):
            # 디렉토리만 들어온 경우도 처리
            if out_path and os.path.isdir(out_path):
                base_dir = out_path
                stem = Path(in_path).stem
            else:
                base_dir = str(Path(in_path).parent)
                stem = Path(in_path).stem
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(Path(base_dir, f"{stem}_result_{ts}.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if stride > 1 and (idx % stride != 0):
                    writer.write(frame)
                    idx += 1
                    continue

                res = self.infer_image(frame, return_vis=True, **kwargs)
                vis_jpg = res["visualization_jpeg"]
                vis = cv2.imdecode(np.frombuffer(vis_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                writer.write(vis)
                idx += 1
        finally:
            cap.release()
            writer.release()

        return out_path


# ========================
# 로컬 테스트
# ========================

if __name__ == "__main__":
    svc = YoloInferenceService(weights=DEFAULT_MODEL_PATH, device=DEFAULT_DEVICE, force_reload=False)

    # 1) 이미지 테스트
    img_path = "samples/test.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        out = svc.infer_image(img, return_vis=True)
        print(f"Detections: {len(out['detections'])}, time_ms={out['time_ms']}")
        Path("samples").mkdir(parents=True, exist_ok=True)
        with open("samples/result.jpg", "wb") as f:
            f.write(out["visualization_jpeg"])
        print("saved -> samples/result.jpg")
    else:
        print(f"[INFO] skip image test: {img_path} not found")

    # 2) 동영상 테스트
    vid_path = r"C:\Users\smhrd1\Desktop\final_project11\sm_final_project\ml_services\yolo_service\samples\insect_8.mp4"
    if os.path.exists(vid_path):
        # out_path를 생략하면 입력 경로 기준으로 _result_YYYYmmdd_HHMMSS.mp4 자동 저장
        out_mp4 = svc.render_video_file(vid_path, stride=1)
        print(f"saved -> {out_mp4}")
    else:
        print(f"[INFO] skip video test: {vid_path} not found")
