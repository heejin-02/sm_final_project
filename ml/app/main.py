"""
리팩토링된 FastAPI 메인 앱
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routers import rag_router, gpt_router, phone_router, upload_router
from app.core.config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# FastAPI 앱 초기화
app = FastAPI(
    title="벌레잡는 109 통합 ML API",
    description="YOLOv5 해충 탐지 + GPT 요약 + RAG 챗봇 통합 서버 (리팩토링 버전)",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(rag_router.router, prefix="/api", tags=["RAG"])
app.include_router(gpt_router.router, prefix="/api", tags=["GPT"])
app.include_router(phone_router.router, prefix="/api", tags=["Phone"])
app.include_router(upload_router.router, prefix="/api", tags=["Upload"])

@app.get("/")
async def root():
    return {
        "service": "벌레잡는 109 통합 ML API",
        "status": "running",
        "version": "2.0.0",
        "architecture": "Clean Architecture",
        "endpoints": [
            "/api/ask",
            "/api/chat", 
            "/api/summary-by-imgidx",
            "/api/daily-gpt-summary",
            "/api/monthly-gpt-summary",
            "/api/yearly-gpt-summary",
            "/api/get-phone",
            "/api/make-call",
            "/api/call-history",
            "/api/upload",
            "/api/signalwire-call",
            "/api/signalwire/voice"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.FASTAPI_HOST, port=settings.FASTAPI_PORT)