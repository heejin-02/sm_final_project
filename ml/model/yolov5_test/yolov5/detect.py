import argparse
import os
import time
from pathlib import Path
from datetime import datetime
import torch
import requests
import cv2
from twilio.rest import Client
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_imshow, check_requirements,
    increment_path, non_max_suppression,
    scale_boxes
)
from utils.torch_utils import select_device, smart_inference_mode
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
# 고정 GH_IDX
gh_idx = 1

# Twilio 설정
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")  # 수신자
PUBLIC_FASTAPI_BASE = "https://d88304504349.ngrok-free.app"

def make_call(insect_name: str, confidence: float):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    url = f"{PUBLIC_FASTAPI_BASE}/twilio-call"
    call = client.calls.create(
        to=USER_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        url=url
    )
    print(f"[전화 발신] Call SID: {call.sid}")

def get_insect_idx(name):
    return {
        "꽃노랑총채벌레": 1,
        "담배가루이": 2,
        "비단노린재": 3,
        "알락수염노린재": 4
    }.get(name, 0)

# 🐛 탐지 결과 API 전송
def send_detection_to_api(insect_name, confidence, img_idx):
    now = datetime.now()
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "anlsModel": "YOLOv5",
        "anlsContent": f"{insect_name} {confidence * 100:.2f}%로 탐지완료",
        "anlsResult": insect_name,
        "createdAt": created_at,
        "insectIdx": get_insect_idx(insect_name),
        "imgIdx": img_idx,
        "notiCheck": 'N',
        "ghIdx": gh_idx,
        "anlsAcc": int(confidence * 100)
    }

    try:
        res = requests.post("http://localhost:8095/api/qc-classification", json=payload)
        print(f"[전송] {insect_name} 저장 완료 | 신뢰도: {confidence:.2f} | 상태코드: {res.status_code}")
    except Exception as e:
        print("[전송 실패]", e)

# 🎥 영상 업로드 함수
def upload_video(file_path, class_id, gh_idx):
    url = "http://localhost:8095/api/qc-videos"
    files = {"video": open(file_path, "rb")}
    data = {"classId": class_id, "ghIdx": gh_idx}
    try:
        res = requests.post(url, files=files, data=data)
        print(f"[서버 응답 상태코드] {res.status_code}")
        print(f"[서버 응답 본문] {res.text}")
        if res.status_code == 200:
            json_res = res.json()
            return json_res.get("imgIdx")
    except Exception as e:
        print("[영상 전송 에러]", e)
    return None

@smart_inference_mode()
def run(weights=Path("best_clean.pt"), source=0, data=Path("data/coco128.yaml"), imgsz=(640, 640),
        conf_thres=0.25, iou_thres=0.45, device="", view_img=True):

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz))

    save_dir = Path("clips")
    save_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame_buffer = []
    recording = False
    start_time = None
    fps = 30
    duration = 10
    insect_name = ""
    best_conf = 0
    video_path = ""

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device).float() / 255.0
        if im.ndim == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        im0 = im0s[0].copy()
        annotator = Annotator(im0, line_width=3, example=str(names))

        detected = False

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    cls_id = int(cls)
                    insect_name = names[cls_id]
                    confidence = float(conf)
                    label = f"{insect_name} {confidence:.2f}"
                    annotator.box_label(xyxy, label, color=colors(cls_id, True))
                    detected = True

                    if not recording:
                        start_time = time.time()
                        video_name = f"{insect_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        video_path = str(save_dir / video_name)
                        out = cv2.VideoWriter(video_path, fourcc, fps, (im0.shape[1], im0.shape[0]))
                        print(f"[녹화 시작] {video_path} | 탐지된 벌레: {insect_name} | 신뢰도: {confidence:.2f}")
                        best_conf = confidence
                        recording = True

        annotated_frame = annotator.result()

        if recording:
            out.write(annotated_frame)
            if time.time() - start_time > duration:
                recording = False
                out.release()
                print("[녹화 종료]")

                converted_path = video_path.replace(".mp4", "_h264.mp4")
                os.system(f'ffmpeg -y -i "{video_path}" -vcodec libx264 -acodec aac "{converted_path}"')
                os.remove(video_path)
                video_path = converted_path

                class_id = get_insect_idx(insect_name)
                img_idx = upload_video(video_path, class_id, gh_idx)
                if img_idx:
                    time.sleep(1)
                    send_detection_to_api(insect_name, best_conf, img_idx)
                    make_call(insect_name, best_conf)

        if view_img:
            cv2.imshow("YOLOv5", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break

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
