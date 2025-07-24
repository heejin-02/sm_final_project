# detect.py - YOLOv5 + Spring Boot 연동 + 대표 이미지 1장만 등록 후 분석 결과 저장

import argparse
import os
from pathlib import Path
import torch
import requests
from datetime import datetime

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    check_img_size, check_imshow, check_requirements,
    cv2, increment_path, non_max_suppression,
    scale_boxes
)
from utils.torch_utils import select_device, smart_inference_mode

# 벌레 이름 → INSECT_IDX 매핑
def get_insect_idx(name):
    return {
        "꽃노랑총채벌레": 1,
        "담배가루이": 2,
        "비단노린재": 3,
        "알락수염노린재": 4
    }.get(name, 0)

# 이미지 업로드 함수
def upload_image(file_path):
    url = "http://localhost:8095/api/qc-images"
    files = {"image": open(file_path, "rb")}
    try:
        res = requests.post(url, files=files)
        if res.status_code == 200:
            img_idx = res.json().get("imgIdx")
            print(f"[업로드 성공] 이미지 등록됨 : {img_idx}")
            return img_idx
        else:
            print(f"[업로드 실패] 실패코드 : {res.status_code}")
    except Exception as e:
        print("[이미지 전송 에러]", e)
    return None

# 탐지 결과 API 전송 함수
def send_detection_to_api(insect_name, confidence, img_idx):
    now = datetime.now()
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "anlsModel": "YOLOv5",
        "anlsConten": "실시간 탐지",
        "anlsResult": insect_name,
        "createdAt": created_at,
        "insectIdx": get_insect_idx(insect_name),
        "imgIdx": img_idx
    }
    try:
        res = requests.post("http://localhost:8095/api/qc-classification", json=payload)
        print(f"[전송] {insect_name} 저장 완료 | 신뢰도: {confidence:.2f} | 상태코드: {res.status_code} | 분석일시: {created_at}")
    except Exception as e:
        print("[전송 실패]", e)

@smart_inference_mode()
def run(weights=Path("best_clean.pt"), source=0, data=Path("data/coco128.yaml"), imgsz=(640, 640),
        conf_thres=0.25, iou_thres=0.45, device="", view_img=True):

    source = str(source)
    webcam = source.isnumeric()
    save_dir = increment_path(Path("runs/detect") / "exp", exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz))

    best_confidence = 0
    best_image = None
    best_insect = None
    last_detection_time = {}

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device).float() / 255.0
        if im.ndim == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    confidence = float(conf)
                    insect_name = names[int(cls)]

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_insect = insect_name
                        best_image = im0.copy()

                    label = f"{insect_name} {confidence:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        if best_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"saved_frames/{best_insect}_{timestamp}.jpg"
            os.makedirs("saved_frames", exist_ok=True)
            cv2.imwrite(save_path, best_image)

            img_idx = upload_image(save_path)
            if img_idx:
                send_detection_to_api(best_insect, best_confidence, img_idx)

            if view_img:
                cv2.imshow("Detection", best_image)
                if cv2.waitKey(1) == ord('q'):
                    return

            best_confidence = 0
            best_image = None
            best_insect = None


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_clean.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
