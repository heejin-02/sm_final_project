import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parent
sys.path.append(str(ROOT))

# YOLOv5 내부 유틸 함수 import
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import save_one_box

# 모델 불러오기 (PosixPath 문제 없음)
model = attempt_load(str(ROOT / 'best.pt'), device='cpu')
model.eval()
names = model.names

# 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 추론
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # 결과 시각화
    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

    # 출력
    cv2.imshow('YOLOv5 Webcam Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
