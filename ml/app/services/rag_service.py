"""
RAG (Retrieval Augmented Generation) 서비스
"""
import logging
from typing import Optional, Tuple
from collections import Counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from app.core.dependencies import get_chat_openai, get_vectorstore
from app.repositories.spring_repository import SpringBootRepository
from app.models.schemas import ChatResponse, SummaryResponse

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.chat = get_chat_openai()
        self.vectorstore = get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.spring_repo = SpringBootRepository()
        
        # 해충 방제 프롬프트
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
        
        # 일반 질문 프롬프트
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
    
    def answer_question(self, question: str) -> ChatResponse:
        """일반 질문에 대한 RAG 답변"""
        try:
            response = self.question_rag_chain.invoke({"input": question})
            return ChatResponse(answer=response["answer"])
            
        except Exception as e:
            logger.error(f"RAG 질문 처리 실패: {e}")
            return ChatResponse(answer="질문 처리 중 오류가 발생했습니다.")
    
    def chat_with_insect_context(self, insect: str, question: str) -> ChatResponse:
        """해충 컨텍스트와 함께 대화"""
        try:
            # 해충 분석 데이터 가져오기
            analysis_text, most_location = self._get_aggregated_analysis_text(insect)
            
            # RAG 체인 실행
            response = self.rag_chain.invoke({
                "insect": insect,
                "most_location": most_location,
                "input": f"{analysis_text}\n\n사용자 질문: {question}"
            })
            
            return ChatResponse(answer=response["answer"])
            
        except Exception as e:
            logger.error(f"해충 컨텍스트 대화 실패: {e}")
            return ChatResponse(answer="대화 처리 중 오류가 발생했습니다.")
    
    def get_summary_by_imgidx(self, img_idx: int) -> SummaryResponse:
        """이미지 인덱스로 GPT 요약 생성"""
        try:
            # Spring Boot로 해충 정보 조회
            result = self.spring_repo.get_summary_by_imgidx(img_idx)
            
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
            
            # 종합 분석 텍스트 생성
            analysis_text, most_location = self._get_aggregated_analysis_text(insect_name)
            
            # RAG 체인으로 요약 생성
            response = self.rag_chain.invoke({
                "insect": insect_name,
                "most_location": most_location,
                "input": analysis_text
            })
            
            # GPT 응답 저장
            if anls_idx:
                self.spring_repo.insert_gpt_summary(
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
            return SummaryResponse(
                status="error",
                summary="요약 생성 중 오류가 발생했습니다."
            )
    
    def _get_aggregated_analysis_text(self, insect_name: str) -> Tuple[str, str]:
        """해충 종합 분석 텍스트 생성"""
        try:
            # Spring Boot API로 분석 데이터 조회
            data_list = self.spring_repo.get_aggregated_analysis_data(insect_name)
            
            if not data_list:
                return "최근 3일간 탐지된 이력이 없습니다.", "정보 없음"
            
            # 위치별 집계
            locations = [item.get("ghName", "알 수 없음") for item in data_list]
            most_common_location, loc_count = Counter(locations).most_common(1)[0]
            
            # 평균 신뢰도 계산
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