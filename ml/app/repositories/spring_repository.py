"""
Spring Boot API 호출을 담당하는 Repository
"""
import requests
import logging
from typing import Optional, Dict, Any, List
from app.core.config import settings

logger = logging.getLogger(__name__)

class SpringBootRepository:
    def __init__(self):
        self.base_url = settings.SPRING_BOOT_URL
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Spring Boot API 요청 공통 처리"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, timeout=30, **kwargs)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Spring API 요청 실패: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Spring API 요청 중 오류: {e}")
            return None
    
    # 사용자 전화번호 조회
    def get_user_phone_by_gh_idx(self, gh_idx: int) -> Optional[str]:
        """온실 인덱스로 사용자 전화번호 조회"""
        result = self._make_request("GET", f"/ml/user-phone-by-ghidx", params={"gh_idx": gh_idx})
        return result.get("phone") if result else None
    
    # 해충 분석 데이터 조회
    def get_aggregated_analysis_data(self, insect_name: str) -> List[Dict[str, Any]]:
        """해충 종합 분석 데이터 조회"""
        result = self._make_request("GET", f"/ml/aggregated-analysis-text", params={"insectName": insect_name})
        return result.get("data", []) if result else []
    
    # GPT 요약 저장
    def insert_gpt_summary(self, anls_idx: int, user_qes: str, gpt_content: str) -> bool:
        """GPT 응답 저장"""
        data = {
            "anlsIdx": anls_idx,
            "userQes": user_qes,
            "gptContent": gpt_content
        }
        result = self._make_request("POST", "/ml/gpt-summary", json=data)
        return result is not None
    
    # 이미지 인덱스로 요약 조회
    def get_summary_by_imgidx(self, img_idx: int) -> Optional[Dict[str, Any]]:
        """이미지 인덱스로 해충 정보 조회"""
        return self._make_request("GET", f"/ml/summary-by-imgidx", params={"imgIdx": img_idx})
    
    # 오늘 탐지 요약 조회  
    def get_today_detection_summary(self) -> List[Dict[str, Any]]:
        """오늘의 탐지 요약 조회"""
        result = self._make_request("GET", "/ml/today-detection-summary")
        return result.get("data", []) if result else []
    
    # 대시보드 요약 저장
    def upsert_dashboard_summary(self, anls_idx: int, summary: str) -> bool:
        """대시보드 요약 저장/업데이트"""
        data = {
            "anlsIdx": anls_idx,
            "summary": summary
        }
        result = self._make_request("POST", "/ml/dashboard-summary", json=data)
        return result is not None
    
    # 리포트 API 호출들
    def get_daily_stats(self, farm_idx: int, date: str) -> Optional[Dict[str, Any]]:
        """일간 통계 조회"""
        return self._make_request("GET", "/report/daily-stats", params={"farmIdx": farm_idx, "date": date})
    
    def get_monthly_stats(self, farm_idx: int, month: str) -> Optional[Dict[str, Any]]:
        """월간 통계 조회"""
        return self._make_request("GET", "/report/monthly-stats", params={"farmIdx": farm_idx, "month": month})
    
    def get_yearly_stats(self, farm_idx: int, year: str) -> Optional[Dict[str, Any]]:
        """연간 통계 조회"""
        return self._make_request("GET", "/report/yearly-stats", params={"farmIdx": farm_idx, "year": year})
    
    # 비디오 업로드
    def upload_video(self, file_content: bytes, filename: str, class_id: int, gh_idx: int) -> Optional[Dict[str, Any]]:
        """Spring Boot로 비디오 업로드"""
        try:
            files = {"video": (filename, file_content)}
            data = {"classId": class_id, "ghIdx": gh_idx}
            
            response = requests.post(
                f"{self.base_url}/api/qc-videos",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"비디오 업로드 실패: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"비디오 업로드 중 오류: {e}")
            return None