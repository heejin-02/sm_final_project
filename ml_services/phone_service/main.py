"""
전화 알림 서비스 - SignalWire 기반 음성 알림
Port: 8005
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import logging
from datetime import datetime
from typing import Optional, List, Dict
import requests
from requests.auth import HTTPBasicAuth

from shared.config import settings
from shared.models import PhoneResponse
from shared.spring_client import SpringBootClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="전화 알림 서비스",
    description="SignalWire 기반 음성 전화 알림 서비스",
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

class CallService:
    def __init__(self):
        logger.info("전화 알림 서비스 초기화 시작...")
        self.spring_client = SpringBootClient()
        self.call_history: List[Dict] = []
        
        # SignalWire 설정
        self.project_id = settings.SIGNALWIRE_PROJECT_ID
        self.auth_token = settings.SIGNALWIRE_AUTH_TOKEN
        self.space_url = settings.SIGNALWIRE_SPACE_URL
        self.from_phone = settings.SIGNALWIRE_PHONE_NUMBER
        
        logger.info("전화 알림 서비스 초기화 완료")
    
    def normalize_phone(self, phone: str) -> str:
        """전화번호 정규화 (+82 형식으로 변환)"""
        if not phone:
            return None
        
        # 숫자만 추출
        digits = ''.join(filter(str.isdigit, phone))
        
        # 한국 번호 처리
        if digits.startswith('82'):
            return f'+{digits}'
        elif digits.startswith('010'):
            return f'+82{digits[1:]}'
        elif digits.startswith('10'):
            return f'+82{digits}'
        
        return f'+82{digits}'
    
    def make_call(self, to_phone: str, insect_name: str, confidence: float) -> bool:
        """SignalWire로 전화 걸기"""
        try:
            normalized_phone = self.normalize_phone(to_phone)
            if not normalized_phone:
                logger.error("유효하지 않은 전화번호")
                return False
            
            # TwiML URL 생성 (실제 서버 URL로 변경 필요)
            twiml_url = f"http://your-server.com:8005/api/signalwire/voice?insect={insect_name}&conf={confidence}"
            
            # SignalWire API 호출
            url = f"https://{self.space_url}/api/laml/2010-04-01/Accounts/{self.project_id}/Calls.json"
            
            data = {
                'To': normalized_phone,
                'From': self.from_phone,
                'Url': twiml_url
            }
            
            response = requests.post(
                url,
                auth=HTTPBasicAuth(self.project_id, self.auth_token),
                data=data
            )
            
            if response.status_code == 201:
                # 통화 기록 저장
                self.call_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'to': normalized_phone,
                    'insect': insect_name,
                    'confidence': confidence,
                    'status': 'success'
                })
                logger.info(f"전화 발신 성공: {normalized_phone}")
                return True
            else:
                logger.error(f"전화 발신 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"전화 발신 중 오류: {e}")
            self.call_history.append({
                'timestamp': datetime.now().isoformat(),
                'to': to_phone,
                'insect': insect_name,
                'confidence': confidence,
                'status': 'failed',
                'error': str(e)
            })
            return False
    
    def make_call_by_gh_idx(self, gh_idx: int, insect_name: str, confidence: float) -> bool:
        """온실 인덱스로 전화 걸기"""
        phone = self.spring_client.get_user_phone_by_gh_idx(gh_idx)
        if phone:
            return self.make_call(phone, insect_name, confidence)
        else:
            logger.error(f"GH_IDX {gh_idx}에 대한 전화번호를 찾을 수 없음")
            return False
    
    def get_call_history(self, limit: int = 10) -> List[Dict]:
        """최근 통화 기록 조회"""
        return self.call_history[-limit:][::-1]

# 서비스 인스턴스
call_service = CallService()

@app.get("/")
async def root():
    return {
        "service": "전화 알림 서비스",
        "status": "running",
        "port": 8005,
        "endpoints": [
            "/api/get-phone",
            "/api/make-call",
            "/api/call-history",
            "/api/signalwire/voice"
        ]
    }

@app.get("/api/get-phone", response_model=PhoneResponse)
async def get_user_phone(gh_idx: int):
    """온실 인덱스로 사용자 전화번호 조회"""
    phone = call_service.spring_client.get_user_phone_by_gh_idx(gh_idx)
    
    if phone:
        return PhoneResponse(phone=phone)
    else:
        return PhoneResponse(message="전화번호 없음")

@app.get("/api/signalwire/voice")
async def signalwire_voice_get(
    insect: str = Query(default="알 수 없는 해충"),
    conf: float = Query(default=None)
):
    """SignalWire TwiML 음성 메시지 생성"""
    logger.info(f"[TwiML] 음성 메시지 생성 요청 - 해충: {insect}, 신뢰도: {conf}")
    
    if conf is not None:
        msg = f"주의하세요. {insect}가 {conf * 100:.1f} 퍼센트 신뢰도로 탐지되었습니다."
    else:
        msg = f"주의하세요. {insect}가 탐지되었습니다."

    logger.info(f"[TwiML] 생성된 음성 메시지: {msg}")

    xml = f"""
    <Response>
        <Say language="ko-KR" voice="alice">{msg}</Say>
    </Response>
    """
    
    logger.info(f"[TwiML] TwiML XML 응답 생성 완료")
    return Response(content=xml.strip(), media_type="application/xml")

@app.post("/api/make-call")
async def make_call(
    gh_idx: int = Query(..., description="온실 인덱스"),
    insect_name: str = Query(default="해충", description="탐지된 해충 이름"),
    confidence: float = Query(default=0.0, description="탐지 신뢰도 (0.0-1.0)")
):
    """SignalWire를 통한 실제 전화 발신"""
    logger.info(f"[API] 전화 발신 요청 수신 - GH_IDX: {gh_idx}, 해충: {insect_name}, 신뢰도: {confidence}")
    
    try:
        success = call_service.make_call_by_gh_idx(gh_idx, insect_name, confidence)
        
        if success:
            logger.info(f"[API] ✅ 전화 발신 성공 - GH_IDX: {gh_idx}")
            return {"message": f"전화 발신 성공 - GH_IDX: {gh_idx}", "status": "success"}
        else:
            logger.error(f"[API] ❌ 전화 발신 실패 - GH_IDX: {gh_idx}")
            raise HTTPException(status_code=500, detail="전화 발신 실패")
            
    except Exception as e:
        logger.error(f"[API] 전화 발신 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/call-history")
async def get_call_history(limit: int = Query(default=10, description="조회할 기록 수")):
    """최근 통화 기록 조회"""
    try:
        history = call_service.get_call_history(limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"통화 기록 조회 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)