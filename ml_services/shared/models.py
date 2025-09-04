"""
공통 Pydantic 모델
"""
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

# Request Models
class QuestionRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    insect: str
    question: str

# Response Models
class ChatResponse(BaseModel):
    answer: str

class SummaryResponse(BaseModel):
    status: str
    summary: str
    anls_idx: Optional[int] = None
    insect: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

class PhoneResponse(BaseModel):
    phone: Optional[str] = None
    message: Optional[str] = None

class UploadResponse(BaseModel):
    videoUrl: str
    imgIdx: int