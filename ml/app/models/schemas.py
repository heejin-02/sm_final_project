"""
Pydantic 스키마 정의
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# 요청 스키마
class QuestionRequest(BaseModel):
    question: str

class InsectRequest(BaseModel):
    insect_name: str
    
class ChatRequest(BaseModel):
    insect: str
    question: str

# 응답 스키마
class ApiResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    
class PhoneResponse(BaseModel):
    phone: Optional[str] = None
    message: Optional[str] = None

class SummaryResponse(BaseModel):
    status: str
    summary: str
    raw_data: Optional[Dict[str, Any]] = None
    anls_idx: Optional[int] = None
    insect: Optional[str] = None

class UploadResponse(BaseModel):
    videoUrl: str
    imgIdx: int

class VideoMetadataResponse(BaseModel):
    videoUrl: str
    imgIdx: int
    classId: int
    insectName: str
    date: str
    time: str
    folder: str

# 내부 데이터 모델
class DetectionData(BaseModel):
    gh_name: str
    insect_name: str
    count: int

class InsectInfo(BaseModel):
    idx: int
    name: str
    
class DailyStatsData(BaseModel):
    totalCount: int
    topZone: Optional[str]
    insectDistribution: List[Dict[str, Any]]
    hourlyStats: List[Dict[str, Any]]
    details: List[Dict[str, Any]]