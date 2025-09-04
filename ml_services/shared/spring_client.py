"""
Spring Boot API 클라이언트
"""
import logging
import requests
from typing import Dict, Any, Optional, List
from .config import settings

logger = logging.getLogger(__name__)

class SpringBootClient:
    def __init__(self):
        self.base_url = settings.SPRING_BOOT_URL
        self.timeout = 10
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """공통 HTTP 요청 처리"""
        url = f"{self.base_url}{endpoint}"
        try:
            kwargs.setdefault('timeout', self.timeout)
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Spring Boot API 요청 실패: {e}")
            return None
    
    def get_user_phone_by_gh_idx(self, gh_idx: int) -> Optional[str]:
        """온실 인덱스로 전화번호 조회"""
        result = self._make_request('GET', f'/api/phone/{gh_idx}')
        if result and 'userPhone' in result:
            return result['userPhone']
        return None
    
    def get_aggregated_analysis_data(self, insect_name: str) -> List[Dict]:
        """해충 종합 분석 데이터 조회"""
        result = self._make_request('GET', '/api/alert/aggregated', params={'insectName': insect_name})
        return result if result else []
    
    def get_summary_by_imgidx(self, img_idx: int) -> Optional[Dict]:
        """이미지 인덱스로 요약 정보 조회"""
        return self._make_request('GET', f'/api/analysis/summary/{img_idx}')
    
    def insert_gpt_summary(self, anls_idx: int, user_qes: str, gpt_content: str) -> bool:
        """GPT 요약 저장"""
        data = {
            'anlsIdx': anls_idx,
            'userQes': user_qes,
            'gptContent': gpt_content
        }
        result = self._make_request('POST', '/api/gpt/save', json=data)
        return result is not None
    
    def get_daily_stats(self, farm_idx: int, date: str) -> Optional[Dict]:
        """일간 통계 조회"""
        params = {'farmIdx': farm_idx, 'date': date}
        return self._make_request('GET', '/report/daily-stats', params=params)
    
    def get_monthly_stats(self, farm_idx: int, month: str) -> Optional[Dict]:
        """월간 통계 조회"""
        params = {'farmIdx': farm_idx, 'month': month}
        return self._make_request('GET', '/report/monthly-stats', params=params)
    
    def get_yearly_stats(self, farm_idx: int, year: str) -> Optional[Dict]:
        """연간 통계 조회"""
        params = {'farmIdx': farm_idx, 'year': year}
        return self._make_request('GET', '/report/yearly-stats', params=params)
    
    def upload_video(self, file_content: bytes, filename: str, class_id: int, gh_idx: int) -> Optional[Dict]:
        """비디오 업로드"""
        files = {'file': (filename, file_content)}
        data = {'cctvIdx': class_id, 'ghIdx': gh_idx}
        return self._make_request('POST', '/api/ml/upload', files=files, data=data)