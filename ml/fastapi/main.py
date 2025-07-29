from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import shutil
import oracledb
import requests
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import requests 
import time
from fastapi import Query
from fastapi.responses import Response
from fastapi import Request

# FastAPI 초기화
app = FastAPI()

# 🌱 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DB 설정
DB_USER = os.getenv("DB_USER")
DB_PASS =  os.getenv("DB_PASS")
DB_DSN =  os.getenv("DB_DSN")
oracledb.init_oracle_client(lib_dir=None)

# 🌐 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 LangChain 설정
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

# 영상 저장 폴더
VIDEO_DIR = Path(r"C:\Users\smhrd1\Desktop\videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 벌레 ID → 이름 매핑 (선택)
INSECT_NAME_MAP = {
    1: "꽃노랑총채벌레",
    2: "담배가루이",
    3: "비단노린재",
    4: "알락수염노린재"
}


# 📋 요청 스키마
class InsectRequest(BaseModel):
    insect_name: str

# 🐛 최근 탐지 내역 요약 함수
def get_recent_analysis_text(insect_name: str) -> str:
    logger = logging.getLogger("uvicorn.error")
    try:
        time.sleep(1)
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT 
                        TO_CHAR(CREATED_AT, 'YYYY-MM-DD HH24:MI:SS') AS TIME,
                        ANLS_ACC || '%' AS CONFIDENCE,
                        ANLS_RESULT
                    FROM QC_CLASSIFICATION
                    WHERE ANLS_RESULT = :1
                      AND CREATED_AT >= SYSDATE - 3
                    ORDER BY CREATED_AT DESC
                """
                cur.execute(sql, [insect_name])
                rows = cur.fetchall()
                print("[DEBUG] DB 쿼리 결과 개수 : ", len(rows))
                print("[DEBUG] 첫 행 : ", rows[0] if rows else "없음")

                if not rows:
                    return "최근 3일 내 탐지된 기록이 없습니다."

                summary_lines = [
                    f"{time}에 {result}가 {confidence}의 신뢰도로 탐지되었습니다."
                    for time, confidence, result in rows
                ]
                print("[DEBUG] FastAPI → GPT 요약에 넘길 텍스트:", summary_lines)
                return "\n".join(summary_lines)

    except Exception as e:
        logger.error(f"[DB ERROR] {e}")
        return "[DB 오류] 분석 데이터를 불러오는 중 문제가 발생했습니다."

# 탐지 후 비디오 영상 업로드하기 
@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    cctv_idx: int = Form(...)
):
    try:
        # 현재 시간 및 저장 경로 설정
        now = datetime.now()
        folder_name = now.strftime("%Y%m%d")
        folder_path = VIDEO_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        file_path = folder_path / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Spring Boot로 파일 업로드 (멀티파트 POST 요청)
        with open(file_path, "rb") as f:
            files = {"video": (file.filename, f, file.content_type)}
            data = {"classId": cctv_idx}  # 또는 classId -> cctvIdx 맞춰줘야 함

            # Spring Boot API 주소로 요청
            print("[UPLOAD DEBUG] 영상 업로드 요청 중...")
            response = requests.post("http://localhost:8095/api/qc-videos", files=files, data=data)
            print("[UPLOAD DEBUG] 응답:", response.status_code, response.text)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Spring Boot 업로드 실패")

        res_json = response.json()
        img_idx = res_json.get("imgIdx")
        video_url = res_json.get("videoUrl")

        print("[DEBUG] Spring Boot 응답 IMG_IDX:", img_idx)

        return {
            "videoUrl": video_url,
            "imgIdx": img_idx
        }

    except Exception as e:
        print("[FastAPI 오류]", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ 2. 영상 메타데이터 조회 API
@app.get("/api/video/{folder}/{video_name}")
async def video_metadata(video_name: str):
    try:
        parts = video_name.replace(".mp4", "").split("_")
        class_id = int(parts[0])
        folder = parts[1]
        time_raw = parts[2]

        video_path = VIDEO_DIR / folder / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        img_idx, db_class_id = get_img_info_by_filename(video_name)
        if img_idx is None:
            raise HTTPException(status_code=404, detail="IMG_IDX not found")

        insect_name = INSECT_NAME_MAP.get(db_class_id or class_id, "Unknown")
        date_str = datetime.strptime(folder, "%Y%m%d").strftime("%Y-%m-%d")
        time_str = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}"
        video_url = f"http://localhost:8000/videos/{folder}/{video_name}"

        return {
            "videoUrl": video_url,
            "imgIdx": img_idx,
            "classId": db_class_id or class_id,
            "insectName": insect_name,
            "date": date_str,
            "time": time_str,
            "folder": folder
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ 공통 함수
def get_img_info_by_filename(video_name: str):
    try:
        class_id = int(video_name.split("_")[0])
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                sql = "SELECT I.IMG_IDX FROM QC_IMAGES I WHERE I.IMG_NAME = :1"
                cur.execute(sql, [video_name])
                result = cur.fetchone()
                if result:
                    return result[0], class_id
    except Exception as e:
        print("[DB ERROR]", e)

    return None, None


# GH_IDX img_idx에서 가져오기
# @app.get("/get_ghIdx")
# def get_ghIdx(imgIdx: int):
#     try:
#         with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
#             with conn.cursor() as cur:
#                 sql = """
#                 SELECT GH_IDX
#                 FROM QC_IMAGES
#                 WHERE IMG_IDX = :1
#                 """
#                 cur.execute(sql, [imgIdx])
#                 result = cur.fetchone()
#                 if result and result[0] is not None:
#                     return {"ghIdx": result[0]}
#                 else:
#                     return {"ghIdx": None}
#     except Exception as e:
#         return {"error": str(e)}

# Twilio API
# GET 방식 
@app.get("/twilio/voice")
def twilio_voice_get(
    insect: str = Query(default="알 수 없는 해충"),
    conf: float | None = Query(default=None)
):
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

# POST 방식 (Twilio가 호출할 때 사용)
@app.post("/twilio/voice")
async def twilio_voice_post(request: Request):
    form = await request.form()
    insect = form.get("insect", "알 수 없는 해충")
    conf = form.get("conf")

    try:
        conf = float(conf)
        msg = f"주의하세요. {insect}가 {conf * 100:.1f} 퍼센트 신뢰도로 탐지되었습니다."
    except:
        msg = f"주의하세요. {insect}가 탐지되었습니다."

    xml = f"""
    <Response>
        <Say language="ko-KR" voice="alice">{msg}</Say>
    </Response>
    """
    return Response(content=xml.strip(), media_type="application/xml")