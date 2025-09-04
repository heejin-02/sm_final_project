"""
RAG 서비스 - 해충 방제 챗봇
Port: 8003
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Tuple
from collections import Counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from shared.config import settings
from shared.models import QuestionRequest, ChatRequest, ChatResponse, SummaryResponse
from shared.spring_client import SpringBootClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="RAG 서비스",
    description="해충 방제 챗봇 및 질의응답 서비스",
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

class RAGService:
    def __init__(self):
        logger.info("RAG 서비스 초기화 시작...")
        
        # OpenAI 설정
        self.chat = ChatOpenAI(
            model=settings.GPT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Embeddings 설정
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # ChromaDB 설정 (로컬 경로 사용)
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Spring Boot 클라이언트
        self.spring_client = SpringBootClient()
        
        # 프롬프트 템플릿
        self.insect_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "다음은 최근 탐지된 해충 '{insect}'에 대한 기록입니다. "
                "'{most_location}' 위치에서 자주 발견되었으니, 이 구역을 중심으로 해충 방제에 신경 써 주세요.\n\n"
                "{context}\n\n"
                "위의 탐지 기록을 바탕으로 이 해충의 특성과 위험성, 그리고 방제 방법을 자세히 알려주세요. "
                "농사를 짓는 어르신도 쉽게 이해하실 수 있도록 부드러운 존댓말 구어체로 설명해 주세요. "
                "인삿말은 생략하고, 문장은 2~3개 정도로 짧고 명확하게 해주시고, 해당 해충 이름을 꼭 포함해주세요."
            )
        ])
        
        self.question_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "사용자의 질문에 대해 아래 context에 기반하여 답변하라. "
                "만약 context에 답이 없으면 '문서에 관련 정보가 없습니다.' 라고 답하라.\n\n{context}"
            ),
            ("human", "{input}")
        ])
        
        # RAG 체인 구성
        self.document_chain = create_stuff_documents_chain(self.chat, self.insect_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.document_chain)
        
        self.question_document_chain = create_stuff_documents_chain(self.chat, self.question_prompt)
        self.question_rag_chain = create_retrieval_chain(self.retriever, self.question_document_chain)
        
        logger.info("RAG 서비스 초기화 완료")
    
    def _get_aggregated_analysis_text(self, insect_name: str) -> Tuple[str, str]:
        """해충 종합 분석 텍스트 생성"""
        try:
            data_list = self.spring_client.get_aggregated_analysis_data(insect_name)
            
            if not data_list:
                return "최근 3일간 탐지된 이력이 없습니다.", "정보 없음"
            
            locations = [item.get("ghName", "알 수 없음") for item in data_list]
            most_common_location, loc_count = Counter(locations).most_common(1)[0]
            
            confidences = [item.get("anlsAcc", 0) for item in data_list if item.get("anlsAcc")]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            summary = (
                f"최근 3일간 '{insect_name}'는 총 {len(data_list)}회 탐지되었습니다. "
                f"그 중 '{most_common_location}' 위치에서 {loc_count}회 감지되었고, "
                f"평균 신뢰도는 {avg_conf:.1f}%입니다."
            )
            
            return summary, most_common_location
            
        except Exception as e:
            logger.error(f"종합 분석 텍스트 생성 실패: {e}")
            return "분석 데이터를 불러오는 중 문제가 발생했습니다.", "정보 없음"

# 서비스 인스턴스
rag_service = RAGService()

@app.get("/")
async def root():
    return {
        "service": "RAG 서비스",
        "status": "running",
        "port": 8003,
        "endpoints": ["/api/ask", "/api/chat", "/api/summary-by-imgidx"]
    }

@app.post("/api/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """일반 질문에 대한 RAG 답변"""
    try:
        response = rag_service.question_rag_chain.invoke({"input": request.question})
        return ChatResponse(answer=response["answer"])
    except Exception as e:
        logger.error(f"RAG 질문 처리 실패: {e}")
        raise HTTPException(status_code=500, detail="질문 처리 중 오류가 발생했습니다.")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """해충 컨텍스트와 함께 대화"""
    try:
        analysis_text, most_location = rag_service._get_aggregated_analysis_text(request.insect)
        
        response = rag_service.rag_chain.invoke({
            "insect": request.insect,
            "most_location": most_location,
            "input": f"{analysis_text}\n\n사용자 질문: {request.question}"
        })
        
        return ChatResponse(answer=response["answer"])
    except Exception as e:
        logger.error(f"해충 컨텍스트 대화 실패: {e}")
        raise HTTPException(status_code=500, detail="대화 처리 중 오류가 발생했습니다.")

@app.get("/api/summary-by-imgidx", response_model=SummaryResponse)
async def get_summary_by_imgidx(imgIdx: int):
    """이미지 인덱스로 GPT 요약 생성"""
    try:
        result = rag_service.spring_client.get_summary_by_imgidx(imgIdx)
        
        if not result:
            return SummaryResponse(
                status="error",
                summary="해당 IMG_IDX에 대한 해충 정보가 없습니다."
            )
        
        insect_name = result.get("insectName")
        anls_idx = result.get("anlsIdx")
        
        if not insect_name:
            return SummaryResponse(
                status="error",
                summary="해충 정보를 찾을 수 없습니다."
            )
        
        analysis_text, most_location = rag_service._get_aggregated_analysis_text(insect_name)
        
        response = rag_service.rag_chain.invoke({
            "insect": insect_name,
            "most_location": most_location,
            "input": analysis_text
        })
        
        if anls_idx:
            rag_service.spring_client.insert_gpt_summary(
                anls_idx=anls_idx,
                user_qes="gpt 응답",
                gpt_content=response["answer"]
            )
        
        return SummaryResponse(
            status="success",
            summary=response["answer"],
            anls_idx=anls_idx,
            insect=insect_name
        )
    except Exception as e:
        logger.error(f"IMG_IDX 요약 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="요약 생성 중 오류가 발생했습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)