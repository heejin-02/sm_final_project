"""
SignalWire를 사용한 전화 발신 서비스
"""
import logging
from typing import Optional
from signalwire.rest import Client
from app.core.config import settings
from app.repositories.spring_repository import SpringBootRepository

logger = logging.getLogger(__name__)

class CallService:
    def __init__(self):
        self.client = None
        self.spring_repo = SpringBootRepository()
        self._initialize_client()
    
    def _initialize_client(self):
        """SignalWire 클라이언트 초기화"""
        try:
            if all([
                settings.SIGNALWIRE_PROJECT_ID,
                settings.SIGNALWIRE_AUTH_TOKEN,
                settings.SIGNALWIRE_SPACE_URL
            ]):
                self.client = Client(
                    settings.SIGNALWIRE_PROJECT_ID,
                    settings.SIGNALWIRE_AUTH_TOKEN,
                    signalwire_space_url=settings.SIGNALWIRE_SPACE_URL
                )
                logger.info("SignalWire 클라이언트 초기화 완료")
            else:
                logger.warning("SignalWire 환경변수 설정이 불완전합니다")
        except Exception as e:
            logger.error(f"SignalWire 클라이언트 초기화 실패: {e}")
    
    def make_call_by_gh_idx(self, gh_idx: int, insect_name: str = "해충", confidence: float = 0.0) -> bool:
        """온실 인덱스로 전화 발신"""
        logger.info(f"[전화발신] 시작 - GH_IDX: {gh_idx}, 해충: {insect_name}, 신뢰도: {confidence}")
        
        try:
            # 1. 전화번호 조회
            logger.info(f"[전화발신] 전화번호 조회 중... GH_IDX: {gh_idx}")
            phone_number = self.spring_repo.get_user_phone_by_gh_idx(gh_idx)
            if not phone_number:
                logger.error(f"[전화발신] 전화번호 조회 실패 - GH_IDX {gh_idx}에 대한 전화번호를 찾을 수 없습니다")
                return False
            
            logger.info(f"[전화발신] 전화번호 조회 성공 - {phone_number}")
            
            # 2. 전화 발신
            logger.info(f"[전화발신] SignalWire 호출 시작")
            return self.make_call(phone_number, insect_name, confidence)
            
        except Exception as e:
            logger.error(f"온실 인덱스 기반 전화 발신 실패: {e}")
            return False
    
    def make_call(self, to_number: str, insect_name: str = "해충", confidence: float = 0.0) -> bool:
        """SignalWire를 통한 실제 전화 발신"""
        logger.info(f"[SignalWire] 전화 발신 준비 - 수신번호: {to_number}")
        
        if not self.client:
            logger.error("[SignalWire] 클라이언트가 초기화되지 않았습니다")
            return False
        
        if not settings.SIGNALWIRE_PHONE_NUMBER:
            logger.error("[SignalWire] 발신자 전화번호가 설정되지 않았습니다")
            return False
        
        try:
            # 전화번호 정규화
            original_number = to_number
            to_number = self._normalize_phone_number(to_number)
            logger.info(f"[SignalWire] 전화번호 정규화: {original_number} -> {to_number}")
            
            # TwiML URL 생성 (ML 서버의 음성 응답 엔드포인트)
            twiml_url = f"http://localhost:{settings.FASTAPI_PORT}/api/signalwire/voice"
            if confidence > 0:
                twiml_url += f"?insect={insect_name}&conf={confidence}"
            else:
                twiml_url += f"?insect={insect_name}"
            
            logger.info(f"[SignalWire] TwiML URL 생성: {twiml_url}")
            logger.info(f"[SignalWire] 발신번호: {settings.SIGNALWIRE_PHONE_NUMBER}")
            
            # 전화 발신
            logger.info(f"[SignalWire] API 호출 중...")
            call = self.client.calls.create(
                url=twiml_url,
                to=to_number,
                from_=settings.SIGNALWIRE_PHONE_NUMBER
            )
            
            logger.info(f"[SignalWire] ✅ 전화 발신 성공!")
            logger.info(f"[SignalWire] 호출 SID: {call.sid}")
            logger.info(f"[SignalWire] 수신번호: {to_number}")
            logger.info(f"[SignalWire] 음성메시지: {insect_name} 탐지됨 (신뢰도: {confidence:.1%})")
            
            return True
            
        except Exception as e:
            logger.error(f"SignalWire 전화 발신 실패: {e}")
            return False
    
    def _normalize_phone_number(self, phone: str) -> str:
        """전화번호를 국제 형식으로 정규화"""
        # 공백, 하이픈 제거
        phone = phone.replace(" ", "").replace("-", "")
        
        # 한국 번호 처리
        if phone.startswith("010"):
            phone = "+82" + phone[1:]  # 010 -> +8210
        elif phone.startswith("82"):
            phone = "+" + phone
        elif not phone.startswith("+"):
            phone = "+82" + phone
            
        return phone
    
    def get_call_history(self, limit: int = 10) -> list:
        """최근 통화 기록 조회"""
        if not self.client:
            logger.error("SignalWire 클라이언트가 초기화되지 않았습니다")
            return []
        
        try:
            calls = self.client.calls.list(limit=limit)
            return [
                {
                    "sid": call.sid,
                    "to": call.to,
                    "from": call.from_,
                    "status": call.status,
                    "start_time": call.start_time,
                    "duration": call.duration
                }
                for call in calls
            ]
        except Exception as e:
            logger.error(f"통화 기록 조회 실패: {e}")
            return []

# 전역 인스턴스
call_service = CallService()