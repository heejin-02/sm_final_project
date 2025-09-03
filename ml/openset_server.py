#!/usr/bin/env python3
"""
Open Set Recognition 실시간 서버
한 번 실행하면 계속 대기하며 요청 처리
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional
from open_set_recognition import ImprovedOpenSetRecognizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="Open Set Pest Detection API", version="1.0.0")

# 전역 변수로 모델 한 번만 로드 (서버 시작 시)
recognizer = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global recognizer
    logger.info("🚀 서버 시작 중...")
    logger.info("📦 모델 로드 중... (첫 실행 시 시간이 걸립니다)")
    
    recognizer = ImprovedOpenSetRecognizer(
        model_path='improved_pest_detection_model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("✅ 모델 로드 완료! 서버 준비됨")
    logger.info("🌐 API 문서: http://localhost:8002/docs")

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "running",
        "service": "Open Set Pest Detection",
        "model_loaded": recognizer is not None
    }

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """파일 업로드로 예측"""
    try:
        # 이미지 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 예측
        result = recognizer.predict_single(image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"예측 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/base64")
async def predict_base64(data: dict):
    """Base64 인코딩된 이미지 예측"""
    try:
        # Base64 디코딩
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 예측
        result = recognizer.predict_single(image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"예측 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/numpy")
async def predict_numpy(data: dict):
    """Numpy 배열로 예측 (라즈베리파이용)"""
    try:
        # Numpy 배열 복원
        image_array = np.array(data['image'], dtype=np.uint8)
        
        # 예측
        result = recognizer.predict_single(image_array)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"예측 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """서버 상태 확인"""
    if recognizer:
        return {
            "status": "ready",
            "known_classes": recognizer.known_insects,
            "thresholds": {
                "max_prob": recognizer.thresholds['max_prob'],
                "entropy": recognizer.thresholds['entropy'],
                "min_distance": recognizer.thresholds['min_distance']
            }
        }
    else:
        return {"status": "model_not_loaded"}

if __name__ == "__main__":
    # 서버 실행 (한 번만 실행하면 계속 동작)
    uvicorn.run(
        app, 
        host="0.0.0.0",  # 모든 IP에서 접속 가능
        port=8002,        # 포트 8002
        reload=False      # 프로덕션에서는 False
    )