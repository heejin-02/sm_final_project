from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
import shutil
import oracledb

# DB 및 FastAPI 초기화
app = FastAPI()
DB_USER = "joo"
DB_PASS = "smhrd4"
DB_DSN = "project-db-campus.smhrd.com:1523/xe"
oracledb.init_oracle_client(lib_dir=None)

# 공통 설정
VIDEO_DIR = Path(r"C:\Users\smhrd1\Desktop\videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 🔻 벌레 이름 매핑
INSECT_NAME_MAP = {
    1: "꽃노랑총채벌레",
    2: "담배가루이",
    3: "비단노린재",
    4: "알락수염노린재"
}

# ✅ 1. 영상 업로드 API
@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    cctv_idx: int = Form(...)
):
    try:
        now = datetime.now()
        folder_name = now.strftime("%Y%m%d")
        folder_path = VIDEO_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        file_path = folder_path / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN) as conn:
            with conn.cursor() as cur:
                img_size = file_path.stat().st_size
                img_ext = file.filename.split(".")[-1]
                created_at = now.strftime("%Y-%m-%d %H:%M:%S")

                cur.execute("""
                    INSERT INTO QC_IMAGES (
                        IMG_IDX, CCTV_IDX, IMG_NAME, IMG_SIZE, IMG_EXT, CREATED_AT
                    ) VALUES (
                        QC_IMAGES_SEQ.NEXTVAL, :1, :2, :3, :4, TO_TIMESTAMP(:5, 'YYYY-MM-DD HH24:MI:SS')
                    ) RETURNING IMG_IDX INTO :6
                """, [cctv_idx, file.filename, img_size, img_ext, created_at, cur.var(int)])
                
                img_idx = cur.getimplicitresults()[0][0]
                conn.commit()

        return {
            "videoUrl": f"/{folder_name}/{file.filename}",
            "imgIdx": img_idx
        }

    except Exception as e:
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
