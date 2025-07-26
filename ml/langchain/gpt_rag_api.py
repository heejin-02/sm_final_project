from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import oracledb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DB 설정
DB_USER = "joo"
DB_PASS = "smhrd4"
DB_DSN = "project-db-campus.smhrd.com:1523/xe"
oracledb.init_oracle_client(lib_dir=None)

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "다음은 최근 탐지 기록이다. 문서와 이 탐지 기록을 종합해, 해당 해충의 위험성과 방제 방법을 사용자에게 친절히 설명하라. "
        "탐지 기록이 없으면 최근에 탐지된 이력이 없다고 설명하라.\n\n{context}"
    )
])

document_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 요청 스키마
class InsectRequest(BaseModel):
    insect_name: str

# 최근 분석 내용 불러오기
def get_recent_analysis_text(insect_name: str):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT 
                        TO_CHAR(CREATED_AT, 'YYYY-MM-DD HH24:MI:SS') AS TIME,
                        SUBSTR(ANLS_CONTENT,
                            INSTR(ANLS_CONTENT, ' ') + 1,
                            INSTR(ANLS_CONTENT, '%') - INSTR(ANLS_CONTENT, ' ') - 1
                        ) || '%' AS CONFIDENCE,
                        ANLS_RESULT
                    FROM QC_CLASSIFICATION
                    WHERE ANLS_RESULT = :1
                      AND CREATED_AT >= SYSDATE - 3
                    ORDER BY CREATED_AT DESC
                """
                cur.execute(sql, [insect_name])
                rows = cur.fetchall()
                if not rows:
                    return "최근 3일 내 탐지된 기록이 없습니다."
                
                summary = ""
                for time, confidence, result in rows:
                    summary += f"{time}에 {result}가 {confidence}의 신뢰도로 탐지되었습니다.\n"
                return summary.strip()
    except Exception as e:
        return f"[DB 오류] 분석 데이터를 불러오는 중 문제가 발생했습니다: {e}"

# 📌 API: 해충 요약 정보 제공
@app.post("/summary")
async def get_insect_summary(data: InsectRequest):
    insect_name = data.insect_name

    # 1. DB에서 최근 탐지 내용 조회
    analysis_text = get_recent_analysis_text(insect_name)

    # 2. 문서 검색 + 요약 생성
    response = rag_chain.invoke({
        "input": analysis_text
    })

    return {
        "insect_name": insect_name,
        "summary": response["answer"]
    }
