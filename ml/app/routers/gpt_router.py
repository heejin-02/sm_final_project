"""
GPT 요약 관련 라우터
"""
from fastapi import APIRouter
from app.models.schemas import SummaryResponse
from app.services.gpt_service import GPTService

router = APIRouter()
gpt_service = GPTService()

@router.get("/daily-gpt-summary", response_model=SummaryResponse)
async def daily_gpt_summary(farm_idx: int, date: str, gh_idx: int = 74):
    """일간 GPT 요약 생성"""
    return gpt_service.generate_daily_summary(farm_idx, date)

@router.get("/monthly-gpt-summary", response_model=SummaryResponse)
async def monthly_gpt_summary(farm_idx: int, month: str):
    """월간 GPT 요약 생성"""
    return gpt_service.generate_monthly_summary(farm_idx, month)

@router.get("/yearly-gpt-summary", response_model=SummaryResponse)
async def yearly_gpt_summary(farm_idx: int, year: str):
    """연간 GPT 요약 생성"""
    return gpt_service.generate_yearly_summary(farm_idx, year)