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
from collections import Counter
from openai import OpenAI 
import socket




# FastAPI 초기화
app = FastAPI()

# 🌱 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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

# ip 주소 가져오기
def get_host_ip():
    return socket.gethostbyname(socket.gethostname())

HOST_IP = get_host_ip()

# 🧠 LangChain 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
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

class InsectRequest(BaseModel):
    insect_name: str

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

# ✅ 분석 텍스트 요약 함수 (단건)
def get_analysis_text_by_img_idx(img_idx: int) -> str:
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        TO_CHAR(C.CREATED_AT, 'YYYY-MM-DD HH24:MI:SS') AS TIME,
                        C.ANLS_RESULT,
                        C.ANLS_ACC
                    FROM QC_CLASSIFICATION C
                    WHERE C.IMG_IDX = :1
                """, [img_idx])
                result = cur.fetchone()

        if not result:
            return "해당 IMG_IDX에 대한 분석 기록이 없습니다."

        time, result_name, acc = result
        return f"{time}에 {result_name}가 {int(acc)}%의 신뢰도로 탐지되었습니다."

    except Exception as e:
        print("[FastAPI ERROR]", e)
        return "[DB 오류] 분석 데이터를 불러오는 중 문제가 발생했습니다."

# ✅ 종합 분석 텍스트 함수 (최근 3일)
def get_aggregated_analysis_text(insect_name: str) -> str:
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        G.GH_NAME,
                        TO_CHAR(C.CREATED_AT, 'YYYY-MM-DD HH24:MI:SS') AS TIME,
                        C.ANLS_ACC
                    FROM QC_CLASSIFICATION C
                    JOIN QC_IMAGES I ON C.IMG_IDX = I.IMG_IDX
                    JOIN QC_GREENHOUSE G ON I.GH_IDX = G.GH_IDX
                    WHERE C.ANLS_RESULT = :1
                      AND C.CREATED_AT >= SYSDATE - 3
                    ORDER BY C.CREATED_AT DESC
                """, [insect_name])
                rows = cur.fetchall()

        if not rows:
            return "최근 3일간 탐지된 이력이 없습니다."

        locations = [r[0] for r in rows]
        most_common_location, loc_count = Counter(locations).most_common(1)[0]
        avg_conf = sum(r[2] for r in rows) / len(rows)

        summary = (
            f"최근 3일간 '{insect_name}'는 총 {len(rows)}회 탐지되었습니다. "
            f"그 중 '{most_common_location}' 위치에서 {loc_count}회 감지되었고, "
            f"평균 신뢰도는 {avg_conf:.1f}%입니다."
        )
        print(f"[DEBUG] 생성된 문장 : {summary}")
        return summary, most_common_location, insect_name

    except Exception as e:
        print("[FastAPI ERROR]", e)
        return "[DB 오류] 탐지 요약 정보를 불러오는 중 문제가 발생했습니다."

def get_today_detection_summary():
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT G.GH_NAME, N.INSECT_NAME, COUNT(*) AS CNT
                    FROM QC_CLASSIFICATION C
                    JOIN QC_IMAGES I ON C.IMG_IDX = I.IMG_IDX
                    JOIN QC_GREENHOUSE G ON I.GH_IDX = G.GH_IDX
                    JOIN QC_INSECT N ON C.INSECT_IDX = N.INSECT_IDX
                    WHERE TRUNC(I.CREATED_AT) = TRUNC(SYSDATE)
                    GROUP BY G.GH_NAME, N.INSECT_NAME
                    ORDER BY CNT DESC
                """)
                return cur.fetchall()
    except Exception as e:
        print("[DB ERROR]", e)
        return []
    

def build_dashboard_prompt(today_data: list[tuple]) -> str:
    if not today_data:
        return "오늘은 해충이 탐지되지 않았습니다. 안심하셔도 됩니다."

    prompt = "오늘 하루 동안 각 구역에서 탐지된 해충 정보입니다:\n\n"
    for gh_name, insect_name, cnt in today_data:
        prompt += f"- {gh_name}에서 {insect_name}가 {cnt}마리 발견됨\n"

    prompt += (
        "\n위 데이터를 참고해 농장주에게 알려줄 짧은 2~3문장의 요약을 만들어 주세요. "
        "중요한 구역과 해충을 알려주고, 존댓말 구어체로 작성해 주세요."
    )
    return prompt

def create_dashboard_summary(today_data: list[tuple]) -> str:
    prompt = build_dashboard_prompt(today_data)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content

def upsert_dashboard_summary(anls_idx: int, prompt_content: str):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT GPT_IDX FROM QC_GPT
                    WHERE USER_QES = '대시보드요약'
                      AND TRUNC(CREATED_AT) = TRUNC(SYSDATE)
                """)
                existing = cur.fetchone()

                if existing:
                    cur.execute("""
                        UPDATE QC_GPT SET GPT_CONTENT = :1, CREATED_AT = SYSDATE
                        WHERE GPT_IDX = :2
                    """, [prompt_content, existing[0]])
                else:
                    cur.execute("""
                        INSERT INTO QC_GPT (
                            GPT_IDX, USER_QES, GPT_CONTENT, CREATED_AT, ANLS_IDX
                        ) VALUES (
                            QC_GPT_SEQ.NEXTVAL, '대시보드요약', :1, SYSDATE, :2
                        )
                    """, [prompt_content, anls_idx])
                conn.commit()
                print("[DB] 대시보드 요약 저장 완료")

    except Exception as e:
        print("[DB ERROR] 대시보드 요약 업서트 실패:", e)


def insert_gpt_summary(anls_idx:int, user_qes:str, gpt_content:str):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO QC_GPT (
                        GPT_IDX,
                        USER_QES,
                        GPT_CONTENT,
                        CREATED_AT,
                        ANLS_IDX
                    ) VALUES (
                        QC_GPT_SEQ.NEXTVAL,
                        :1,
                        :2,
                        SYSDATE,
                        :3
                    )
                """, [
                    user_qes,
                    gpt_content,
                    anls_idx
                ])
                conn.commit()
                print("[DB] GPT 응답 저장 완료")
    except Exception as e:
        print("[DB ERROR]", e)

# ✅ API: GPT 요약 (imgIdx 기반 → 벌레 전체 요약)
@app.get("/api/summary-by-imgidx")
async def get_summary_by_imgidx(imgIdx: int):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT C.ANLS_RESULT, C.ANLS_IDX
                    FROM QC_CLASSIFICATION C
                    WHERE C.IMG_IDX = :1
                """, [imgIdx])
                result = cur.fetchone()

        if not result:
            return {"status": "error", "message": "해당 IMG_IDX에 대한 해충 정보가 없습니다."}

        insect_name, anls_idx = result
        print(f"[DEBUG] IMG_IDX {imgIdx} → 벌레 이름: {insect_name}")
        
        summary, most_common_location, insect_name = get_aggregated_analysis_text(insect_name)
        print(f"[DEBUG] 가장 많이 나온 장소 : {most_common_location}")
        response = rag_chain.invoke({
        "insect":insect_name,
        "most_location" : most_common_location,
        "input": summary
        
        })

        insert_gpt_summary(
            anls_idx = anls_idx,
            user_qes = "gpt 응답", 
            gpt_content= response["answer"])

           # 2. 대시보드 요약 자동 생성 및 저장
        today_data = get_today_detection_summary()
        dashboard_summary = create_dashboard_summary(today_data)
        upsert_dashboard_summary(anls_idx, prompt_content=dashboard_summary)

        return {
            "status": "success",
            "anls_idx" : anls_idx,
            "insect": insect_name,
            "solution_summary": response["answer"]
        }
    
    except Exception as e:
        print("[FastAPI ERROR]", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
   
import requests

def upsert_daily_report(farm_idx: int, date_str: str, summary: str):
    with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
        with conn.cursor() as cur:
            # 1. 이미 존재하는지 확인
            cur.execute("""
                SELECT REPORT_IDX FROM QC_REPORT
                WHERE FARM_IDX = :1 AND PERIOD_TYPE = '일간' AND PERIOD_MARK = :2
            """, [farm_idx, date_str])
            existing = cur.fetchone()

            if existing:
                # 2. UPDATE
                cur.execute("""
                    UPDATE QC_REPORT
                    SET REPORT = :1, CREATED_AT = SYSTIMESTAMP
                    WHERE REPORT_IDX = :2
                """, [summary, existing[0]])
            else:
                # 3. INSERT
                cur.execute("""
                    INSERT INTO QC_REPORT (REPORT_IDX, FARM_IDX, PERIOD_TYPE, PERIOD_MARK, REPORT, CREATED_AT, GPT_IDX)
                    VALUES (QC_REPORT_SEQ.NEXTVAL, :1, '일간', :2, :3, SYSTIMESTAMP, NULL)
                """, [farm_idx, date_str, summary])
        conn.commit()

def get_farm_name_by_idx(farm_idx: int) -> str:
    with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT FARM_NAME FROM QC_FARM WHERE FARM_IDX = :1", [farm_idx])
            row = cur.fetchone()
            return row[0] if row else "알 수 없는 농장"
        
def build_daily_stats_prompt(data: dict, date: str, farm_name: str) -> str:
    total = data.get("totalCount", 0)
    top_zone = data.get("topZone", "정보 없음")
    insects = data.get("insectDistribution", [])
    hourly = data.get("hourlyStats", [])

    # 가장 많은 해충
    if insects:
        top_insect = max(insects, key=lambda x: x["count"])
        top_insect_name = top_insect["insect"]
        top_insect_ratio = round((top_insect["count"] / total) * 100)
    else:
        top_insect_name = "정보 없음"
        top_insect_ratio = 0

    # 활동량이 많은 시간대
    if hourly:
        top_hour = int(hourly[0]["hour"])
        hour_range = f"{top_hour}시~{top_hour+2}시"
    else:
        hour_range = "정보 없음"

    # 최종 프롬프트
    prompt = (
        f"{date} 기준 {farm_name}의 해충 탐지 요약입니다.\n"
        f"오늘은 총 {total}마리의 해충이 탐지되었고, "
        f"{top_insect_name}가 가장 많은 비중({top_insect_ratio}%)을 차지했어요.\n"
        f"{top_zone}에서 가장 많이 탐지되었고, {hour_range} 사이에 활동량이 높았습니다.\n\n"
        "위 내용을 인사말은 제외하고, 농장주에게 보고하는 2~3문장의 친절한 요약으로 작성해주세요. 존댓말 구어체로 부탁드립니다."
    )
    return prompt

def get_existing_daily_summary(farm_idx: int, date_str: str):
    with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT REPORT FROM QC_REPORT
                WHERE FARM_IDX = :1 AND PERIOD_TYPE = '일간' AND PERIOD_MARK = :2
            """, [farm_idx, date_str])
            row = cur.fetchone()
            return row[0] if row else None

@app.get("/api/daily-gpt-summary")
def gpt_daily_summary(farm_idx: int, date: str):
    try:
        # 1. 기존 요약이 있으면 그대로 반환
        existing_summary = get_existing_daily_summary(farm_idx, date)
        if existing_summary:
            return {
                "status": "already_exists",
                "summary": existing_summary,
                "raw_data": None
            }

        # 2. Spring API 호출
        params = {"farmIdx": farm_idx, "date": date}
        res = requests.get("http://localhost:8095/report/daily-stats", params=params)
        if res.status_code != 200:
            return {"error": f"Spring API 호출 실패: {res.status_code}"}
        data = res.json()

        # 3. 탐지 없으면 GPT 요청 생략
        if not data or data.get("totalCount", 0) == 0 or not data.get("details"):
            return {
                "status": "no_detection",
                "summary": f"{date} 기준으로 {farm_idx}번 농장에는 해충 탐지 정보가 없습니다. 오늘은 안전한 날이에요!",
                "raw_data": data
            }

        # 4. GPT 프롬프트 생성 및 요청
        farm_name = get_farm_name_by_idx(farm_idx)
        prompt = build_daily_stats_prompt(data, date=date, farm_name=farm_name)
        gpt_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        summary = gpt_res.choices[0].message.content

        # 5. DB 저장
        upsert_daily_report(farm_idx, date, summary)

        return {
            "status": "success",
            "summary": summary,
            "raw_data": data
        }

    except Exception as e:
        return {"error": str(e)}



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
        video_url = f"http://{HOST_IP}:8000/videos/{folder}/{video_name}"

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
    
# ✅ 자동 실행용 API: 영상 업로드/탐지 후 호출
@app.post("/api/update-dashboard-summary")
def update_dashboard_summary(anls_idx: int):
    today_data = get_today_detection_summary()
    summary = create_dashboard_summary(today_data)
    upsert_dashboard_summary(anls_idx, prompt_content=summary)
    return {"message": "대시보드 요약이 갱신되었습니다.", "summary": summary}

# ✅ 대시보드 조회 API
@app.get("/api/daily-zone-summary")
def daily_zone_summary():
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT GPT_CONTENT
                    FROM (
                        SELECT GPT_CONTENT
                        FROM QC_GPT
                        WHERE USER_QES = '대시보드요약'
                          AND TRUNC(CREATED_AT) = TRUNC(SYSDATE)
                        ORDER BY CREATED_AT DESC
                    )
                    WHERE ROWNUM = 1
                """)
                row = cur.fetchone()
                content = row[0].read() if row and hasattr(row[0], 'read') else row[0]  # CLOB 처리
        return {
            "summary_text": content if content else "오늘 탐지된 해충이 없습니다."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Twilio/Signalwire API 
@app.get("/api/get-phone")
def get_user_phone(gh_idx: int):
    try:
        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT U.USER_PHONE
                    FROM QC_USER U
                    JOIN QC_FARM F ON U.USER_PHONE = F.USER_PHONE
                    JOIN QC_GREENHOUSE G ON F.FARM_IDX = G.FARM_IDX
                    WHERE G.GH_IDX = :gh_idx
                """, [gh_idx])
                row = cur.fetchone()
                if row:
                    return {"phone": row[0]}
        return JSONResponse(status_code=404, content={"message": "전화번호 없음"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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