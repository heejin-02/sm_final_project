"""
공통 유틸리티 함수들
"""
import re
import socket
from datetime import datetime
from typing import Dict, Optional

def normalize_phone(phone: str) -> str:
    """전화번호를 국제 형식으로 정규화"""
    digits_only = re.sub(r"[^0-9]", "", phone)
    if digits_only.startswith("0"):
        digits_only = digits_only[1:]
    return f"+82{digits_only}"

def get_host_ip() -> str:
    """로컬 IP 주소 가져오기"""
    return socket.gethostbyname(socket.gethostname())

def get_insect_idx(name: str) -> int:
    """해충 이름으로 인덱스 가져오기"""
    return {
        "꽃노랑총채벌레": 1,
        "담배가루이": 2,
        "비단노린재": 3,
        "알락수염노린재": 4
    }.get(name, 0)

def get_insect_name_by_idx(idx: int) -> str:
    """해충 인덱스로 이름 가져오기"""
    return {
        1: "꽃노랑총채벌레",
        2: "담배가루이", 
        3: "비단노린재",
        4: "알락수염노린재"
    }.get(idx, "Unknown")

def format_datetime_for_voice(dt_str: str) -> str:
    """datetime을 음성용으로 포맷팅"""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return f"{dt.month}월 {dt.day}일 {dt.hour}시 {dt.minute}분"
    except:
        return "알 수 없는 시간"

def parse_video_filename(video_name: str) -> Dict[str, any]:
    """비디오 파일명을 파싱하여 메타데이터 추출"""
    try:
        parts = video_name.replace(".mp4", "").split("_")
        class_id = int(parts[0])
        folder = parts[1] 
        time_raw = parts[2]
        
        date_str = datetime.strptime(folder, "%Y%m%d").strftime("%Y-%m-%d")
        time_str = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}"
        
        return {
            "class_id": class_id,
            "folder": folder,
            "date": date_str,
            "time": time_str
        }
    except Exception as e:
        raise ValueError(f"Invalid video filename format: {video_name}")

def build_video_url(host_ip: str, folder: str, video_name: str, port: int = 8000) -> str:
    """비디오 URL 생성"""
    return f"http://{host_ip}:{port}/videos/{folder}/{video_name}"