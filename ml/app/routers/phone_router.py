"""
전화 관련 라우터 
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response, JSONResponse
from app.models.schemas import PhoneResponse
from app.repositories.spring_repository import SpringBootRepository
from app.services.call_service import call_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
spring_repo = SpringBootRepository()

@router.get("/get-phone", response_model=PhoneResponse)
async def get_user_phone(gh_idx: int):
    """온실 인덱스로 사용자 전화번호 조회"""
    phone = spring_repo.get_user_phone_by_gh_idx(gh_idx)
    
    if phone:
        return PhoneResponse(phone=phone)
    else:
        return PhoneResponse(message="전화번호 없음")

@router.get("/signalwire/voice")
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

@router.post("/signalwire-call")
async def signalwire_call():
    """최근 탐지 기록으로 SignalWire 호출 메시지 생성"""
    # Spring Boot API로 최근 탐지 데이터 조회하는 로직 추가 필요
    # 현재는 기본 메시지 반환
    
    msg = "해충이 탐지되었습니다. 확인해 주세요."
    
    twiml = f"""
    <Response>
        <Say language="ko-KR" voice="Polly.Seoyeon">{msg}</Say>
    </Response>
    """
    
    return Response(content=twiml.strip(), media_type="application/xml")

@router.post("/make-call")
async def make_call(
    gh_idx: int = Query(..., description="온실 인덱스"),
    insect_name: str = Query(default="해충", description="탐지된 해충 이름"),
    confidence: float = Query(default=0.0, description="탐지 신뢰도 (0.0-1.0)")
):
    """SignalWire를 통한 실제 전화 발신"""
    logger.info(f"[API] 전화 발신 요청 수신 - GH_IDX: {gh_idx}, 해충: {insect_name}, 신뢰도: {confidence}")
    
    try:
        logger.info(f"[API] CallService 호출 시작")
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

@router.get("/call-history")
async def get_call_history(limit: int = Query(default=10, description="조회할 기록 수")):
    """최근 통화 기록 조회"""
    try:
        history = call_service.get_call_history(limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"통화 기록 조회 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))