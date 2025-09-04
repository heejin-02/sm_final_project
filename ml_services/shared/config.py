"""
공통 환경 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class SharedConfig:
    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GPT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Spring Boot 연동 설정
    SPRING_BOOT_URL: str = os.getenv("SPRING_BOOT_URL", "http://localhost:8095")
    
    # SignalWire 설정
    SIGNALWIRE_PROJECT_ID: str = os.getenv("SIGNALWIRE_PROJECT_ID")
    SIGNALWIRE_AUTH_TOKEN: str = os.getenv("SIGNALWIRE_AUTH_TOKEN")
    SIGNALWIRE_PHONE_NUMBER: str = os.getenv("SIGNALWIRE_PHONE_NUMBER")
    SIGNALWIRE_SPACE_URL: str = os.getenv("SIGNALWIRE_SPACE_URL")
    
    # 파일 경로 설정
    VIDEO_DIR: Path = Path("./videos")
    CHROMA_DB_PATH: str = "./chroma_db"
    
    # 온실 고정값
    DEFAULT_GH_IDX: int = 74

settings = SharedConfig()