from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import shutil
import oracledb
import requests
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# FastAPI 초기화
app = FastAPI()

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_DSN = os.getenv("DB_DSN")
oracledb.init_oracle_client(lib_dir=None)

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
        "사용자가 탐지한 해충은 {insect_name}이고, 신뢰도는 {confidence}%이다.\n"
        "다음은 방금 탐지된 탐지 기록이다:\n\n{input}\n\n"
        "이 해충의 위험성과 방제 방법을 사용자에게 친절하고 쉽게 설명하라."
    )
])

document_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

VIDEO_DIR = Path(r"C:\\Users\\smhrd1\\Desktop\\videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

INSECT_NAME_MAP = {
    1: "꽃노랑총채벌레",
    2: "담배가루이",
    3: "비단노린재",
    4: "알락수염노린재"
}

class InsectRequest(BaseModel):
    insect_name: str

def get_recent_analysis_text(insect_name: str) -> str:
    logger = logging.getLogger("uvicorn.error")
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT TO_CHAR(CREATED_AT, 'YYYY-MM-DD HH24:MI:SS') AS TIME,
                           ANLS_ACC || '%' AS CONFIDENCE,
                           ANLS_RESULT
                    FROM QC_CLASSIFICATION
                    WHERE ANLS_RESULT = :1
                      AND CREATED_AT >= SYSDATE - 3
                    ORDER BY CREATED_AT DESC
                """, [insect_name])
                rows = cur.fetchall()
                if not rows:
                    return "최근 3일 내 탐지된 기록이 없습니다."
                return "\n".join([f"{t}에 {r}가 {c}의 신뢰도로 탐지되었습니다." for t, c, r in rows])
    except Exception as e:
        logger.error(f"[DB ERROR] {e}")
        return "[DB 오류] 분석 데이터를 불러오는 중 문제가 발생했습니다."

def get_classification_info_by_anls_idx(anls_idx: int):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT N.INSECT_NAME, TO_CHAR(C.ANLS_ACC)
                    FROM QC_CLASSIFICATION C
                    JOIN QC_INSECT N ON C.INSECT_IDX = N.INSECT_IDX
                    WHERE C.ANLS_IDX = :1
                """, [anls_idx])
                result = cur.fetchone()
                return result if result else ("알 수 없음", None)
    except Exception as e:
        print("[ANLS_IDX → 분석정보 조회 오류]", e)
        return ("알 수 없음", None)

def save_gpt_response(anls_idx: int, gpt_content: str):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO QC_GPT (GPT_IDX, USER_QES, GPT_CONTENT, CREATED_AT, ANLS_IDX)
                    VALUES (QC_GPT_SEQ.NEXTVAL, :1, :2, SYSDATE, :3)
                """, ["자동 생성 요청", gpt_content, anls_idx])
                conn.commit()
                print("[GPT 응답 저장 완료]")
    except Exception as e:
        print("[GPT 응답 저장 오류]", e)

def generate_summary_by_anls_idx(anls_idx: int):
    insect_name, confidence = get_classification_info_by_anls_idx(anls_idx)
    context = get_recent_analysis_text(insect_name)

    gpt_input = {
        "input": context,
        "insect_name": insect_name,
        "confidence": confidence
    }

    gpt_response = rag_chain.invoke(gpt_input)

    if hasattr(gpt_response, "content"):
        result = gpt_response.content
    elif isinstance(gpt_response, str):
        result = gpt_response
    else:
        result = "[GPT 응답 없음]"

    save_gpt_response(anls_idx, result)
    return result

def get_anls_idx_by_img_idx(img_idx: int) -> int | None:
    try:
        for _ in range(5):  # 최대 5회 재시도
            with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT ANLS_IDX
                        FROM QC_CLASSIFICATION
                        WHERE IMG_IDX = :1
                        ORDER BY CREATED_AT DESC
                    """, [img_idx])
                    result = cur.fetchone()
                    if result:
                        return result[0]
            time.sleep(0.5)
    except Exception as e:
        print("[IMG_IDX → ANLS_IDX 조회 오류]", e)
    return None

@app.get("/api/gpt-summary_view")
def get_saved_gpt_summary(anls_idx: int):
    print(f"📌 [요약 요청] anls_idx: {anls_idx}")
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT TO_CHAR(GPT_CONTENT)
                    FROM QC_GPT
                    WHERE ANLS_IDX = :1
                    ORDER BY CREATED_AT DESC
                """, [anls_idx])
                result = cur.fetchone()
                if result:
                    return {"summary": result[0]}
                else:
                    return {"summary": "해당 분석에 대한 요약이 아직 없습니다."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/api/summary")
def get_gpt_summary(anls_idx: int):
    print(f"🚀 [GPT 요약 생성 요청] anls_idx: {anls_idx}")
    summary = generate_summary_by_anls_idx(anls_idx)
    return {"summary": summary}

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

        if img_idx:
            anls_idx = get_anls_idx_by_img_idx(img_idx)
            if anls_idx:
                generate_summary_by_anls_idx(anls_idx)

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

@app.post("/twilio-call")
def twilio_call():
    with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
        with conn.cursor() as cur:
            # 가장 최근 탐지 기록 1건 가져오기
            cur.execute("""
            SELECT * FROM (
                SELECT I.CREATED_AT, G.GH_NAME, N.INSECT_NAME
                FROM QC_CLASSIFICATION C
                JOIN QC_IMAGES I ON C.IMG_IDX = I.IMG_IDX
                JOIN QC_GREENHOUSE G ON I.GH_IDX = G.GH_IDX
                JOIN QC_INSECT N ON C.INSECT_IDX = N.INSECT_IDX
                ORDER BY I.CREATED_AT DESC
            )
            WHERE ROWNUM = 1
        """)
            row = cur.fetchone()

    if not row:
        msg = "최근 탐지된 해충 정보가 없습니다."
    else:
        _, gh_name, insect_name = row

        msg = f"{gh_name}에서 {insect_name}가 탐지되었습니다. 확인해 주세요."

    # TwiML XML 구성
    twiml = f"""
    <Response>
        <Say language="ko-KR" voice="Polly.Seoyeon">{msg}</Say>
    </Response>
    """

    return Response(content=twiml.strip(), media_type="application/xml")