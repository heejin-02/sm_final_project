"""
파일 업로드 관련 라우터
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.models.schemas import UploadResponse
from app.repositories.spring_repository import SpringBootRepository
from app.core.config import settings

router = APIRouter()
spring_repo = SpringBootRepository()

@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    cctv_idx: int = Form(...)
):
    """비디오 파일 업로드"""
    try:
        # 파일 내용 읽기
        file_content = await file.read()
        
        # Spring Boot로 업로드
        result = spring_repo.upload_video(
            file_content=file_content,
            filename=file.filename,
            class_id=cctv_idx,
            gh_idx=settings.DEFAULT_GH_IDX
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="파일 업로드 실패")
        
        return UploadResponse(
            videoUrl=result.get("videoUrl", ""),
            imgIdx=result.get("imgIdx", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 처리 중 오류: {str(e)}")