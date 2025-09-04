"""
파일 프록시 서비스 - 비디오 업로드 프록시
Port: 8006
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from shared.config import settings
from shared.models import UploadResponse
from shared.spring_client import SpringBootClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="파일 프록시 서비스",
    description="Spring Boot로의 파일 업로드 프록시 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spring Boot 클라이언트
spring_client = SpringBootClient()

@app.get("/")
async def root():
    return {
        "service": "파일 프록시 서비스",
        "status": "running",
        "port": 8006,
        "endpoints": ["/api/upload"]
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    cctv_idx: int = Form(...)
):
    """비디오 파일 업로드 프록시"""
    try:
        logger.info(f"파일 업로드 요청: {file.filename}, CCTV_IDX: {cctv_idx}")
        
        # 파일 내용 읽기
        file_content = await file.read()
        
        # Spring Boot로 업로드
        result = spring_client.upload_video(
            file_content=file_content,
            filename=file.filename,
            class_id=cctv_idx,
            gh_idx=settings.DEFAULT_GH_IDX
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="파일 업로드 실패")
        
        logger.info(f"파일 업로드 성공: {result}")
        
        return UploadResponse(
            videoUrl=result.get("videoUrl", ""),
            imgIdx=result.get("imgIdx", 0)
        )
        
    except Exception as e:
        logger.error(f"업로드 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)