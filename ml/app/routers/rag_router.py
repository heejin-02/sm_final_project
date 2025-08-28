"""
RAG 관련 라우터
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, ChatRequest, ChatResponse, SummaryResponse
from app.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """일반 질문에 대한 RAG 답변"""
    return rag_service.answer_question(request.question)

@router.post("/chat", response_model=ChatResponse)  
async def chat_with_context(request: ChatRequest):
    """해충 컨텍스트와 함께 대화"""
    return rag_service.chat_with_insect_context(request.insect, request.question)

@router.get("/summary-by-imgidx", response_model=SummaryResponse)
async def get_summary_by_imgidx(imgIdx: int):
    """이미지 인덱스로 GPT 요약 생성"""
    return rag_service.get_summary_by_imgidx(imgIdx)