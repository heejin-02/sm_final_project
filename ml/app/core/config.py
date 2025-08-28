"""
환경 변수 및 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Database 설정
    DB_USER: str = os.getenv("DB_USER") 
    DB_PASS: str = os.getenv("DB_PASS")
    DB_DSN: str = os.getenv("DB_DSN")
    
    # Spring Boot 연동 설정
    SPRING_BOOT_URL: str = os.getenv("SPRING_BOOT_URL", "http://localhost:8095")
    
    # FastAPI 서버 설정
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", 8003))
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    
    # SignalWire 설정
    SIGNALWIRE_PROJECT_ID: str = os.getenv("SIGNALWIRE_PROJECT_ID")
    SIGNALWIRE_AUTH_TOKEN: str = os.getenv("SIGNALWIRE_AUTH_TOKEN")
    SIGNALWIRE_PHONE_NUMBER: str = os.getenv("SIGNALWIRE_PHONE_NUMBER")
    SIGNALWIRE_SPACE_URL: str = os.getenv("SIGNALWIRE_SPACE_URL")
    
    # 파일 경로 설정
    VIDEO_DIR: Path = Path("./videos")
    CHROMA_DB_PATH: str = "./chroma_db"
    MODEL_WEIGHTS_PATH: str = "best.pt"
    
    # GPT 모델 설정
    GPT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # 온실 고정값 
    DEFAULT_GH_IDX: int = 74

settings = Settings()