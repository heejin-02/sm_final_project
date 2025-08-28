"""
전화 관련 라우터 
"""
from fastapi import APIRouter, Query
from fastapi.responses import Response, JSONResponse
from app.models.schemas import PhoneResponse
from app.repositories.spring_repository import SpringBootRepository

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

@router.get("/twilio/voice")
async def twilio_voice_get(
    insect: str = Query(default="알 수 없는 해충"),
    conf: float = Query(default=None)
):
    """Twilio 음성 메시지 생성"""
    if conf is not None:
        msg = f"주의하세요. {insect}가 {conf * 100:.1f} 퍼센트 신뢰도로 탐지되었습니다."
    else:
        msg = f"주의하세요. {insect}가 탐지되었습니다."

    xml = f"""
    <Response>
        <Say language="ko-KR" voice="alice">{msg}</Say>
    </Response>
    """
    return Response(content=xml.strip(), media_type="application/xml")

@router.post("/twilio-call")
async def twilio_call():
    """최근 탐지 기록으로 Twilio 호출 메시지 생성"""
    # Spring Boot API로 최근 탐지 데이터 조회하는 로직 추가 필요
    # 현재는 기본 메시지 반환
    
    msg = "해충이 탐지되었습니다. 확인해 주세요."
    
    twiml = f"""
    <Response>
        <Say language="ko-KR" voice="Polly.Seoyeon">{msg}</Say>
    </Response>
    """
    
    return Response(content=twiml.strip(), media_type="application/xml")