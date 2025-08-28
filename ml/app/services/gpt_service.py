"""
GPT 관련 서비스 로직
"""
import logging
from typing import Dict, Any, Optional, List
from collections import Counter
from datetime import datetime
from dateutil.relativedelta import relativedelta

from app.core.dependencies import get_openai_client
from app.repositories.spring_repository import SpringBootRepository
from app.models.schemas import SummaryResponse

logger = logging.getLogger(__name__)

class GPTService:
    def __init__(self):
        self.openai_client = get_openai_client()
        self.spring_repo = SpringBootRepository()
    
    def create_dashboard_summary(self, today_data: List[Dict[str, Any]]) -> str:
        """오늘 탐지 데이터로 대시보드 요약 생성"""
        if not today_data:
            return "오늘은 해충이 탐지되지 않았습니다. 안심하셔도 됩니다."
        
        prompt = "오늘 하루 동안 각 구역에서 탐지된 해충 정보입니다:\n\n"
        for data in today_data:
            gh_name = data.get("ghName", "알 수 없는 구역")
            insect_name = data.get("insectName", "알 수 없는 해충") 
            count = data.get("count", 0)
            prompt += f"- {gh_name}에서 {insect_name}가 {count}마리 발견됨\n"
        
        prompt += (
            "\n위 데이터를 참고해 농장주에게 알려줄 짧은 2~3문장의 요약을 만들어 주세요. "
            "중요한 구역과 해충을 알려주고, 존댓말 구어체로 작성해 주세요."
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT 대시보드 요약 생성 실패: {e}")
            return "요약 생성 중 오류가 발생했습니다."
    
    def generate_daily_summary(self, farm_idx: int, date: str) -> SummaryResponse:
        """일간 GPT 요약 생성"""
        try:
            # Spring API로 일간 통계 조회
            data = self.spring_repo.get_daily_stats(farm_idx, date)
            
            if not data or data.get("totalCount", 0) == 0:
                return SummaryResponse(
                    status="no_detection",
                    summary=f"{date} 기준으로 {farm_idx}번 농장에는 해충 탐지 정보가 없습니다.",
                    raw_data=data
                )
            
            # GPT 프롬프트 생성 및 요청
            prompt = self._build_daily_stats_prompt(data, date, farm_idx)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            
            summary = response.choices[0].message.content
            
            return SummaryResponse(
                status="success",
                summary=summary,
                raw_data=data
            )
            
        except Exception as e:
            logger.error(f"일간 요약 생성 실패: {e}")
            return SummaryResponse(
                status="error", 
                summary="요약 생성 중 오류가 발생했습니다.",
                raw_data=None
            )
    
    def generate_monthly_summary(self, farm_idx: int, month: str) -> SummaryResponse:
        """월간 GPT 요약 생성"""
        try:
            data = self.spring_repo.get_monthly_stats(farm_idx, month)
            
            if not data or data.get("totalCount", 0) == 0:
                return SummaryResponse(
                    status="no_detection",
                    summary=f"{month}월 기준 {farm_idx}번 농장에는 해충 탐지 정보가 없습니다.",
                    raw_data=data
                )
            
            prompt = self._build_monthly_stats_prompt(data, month, farm_idx)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            
            summary = response.choices[0].message.content
            
            return SummaryResponse(
                status="success",
                summary=summary,
                raw_data=data
            )
            
        except Exception as e:
            logger.error(f"월간 요약 생성 실패: {e}")
            return SummaryResponse(
                status="error",
                summary="요약 생성 중 오류가 발생했습니다.",
                raw_data=None
            )
    
    def generate_yearly_summary(self, farm_idx: int, year: str) -> SummaryResponse:
        """연간 GPT 요약 생성"""
        try:
            data = self.spring_repo.get_yearly_stats(farm_idx, year)
            
            if not data or data.get("totalCount", 0) == 0:
                return SummaryResponse(
                    status="no_detection",
                    summary=f"{year} 연간 {farm_idx}번 농장에는 해충 탐지 정보가 없습니다.",
                    raw_data=data
                )
            
            prompt = self._build_yearly_stats_prompt(data, year, farm_idx)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            
            summary = response.choices[0].message.content
            
            return SummaryResponse(
                status="success",
                summary=summary,
                raw_data=data
            )
            
        except Exception as e:
            logger.error(f"연간 요약 생성 실패: {e}")
            return SummaryResponse(
                status="error",
                summary="요약 생성 중 오류가 발생했습니다.",
                raw_data=None
            )
    
    def _build_daily_stats_prompt(self, data: Dict[str, Any], date: str, farm_idx: int) -> str:
        """일간 통계 프롬프트 생성"""
        total = data.get("totalCount", 0)
        top_zone = data.get("topZone", "정보 없음")
        insects = data.get("insectDistribution", [])
        hourly = data.get("hourlyStats", [])
        
        if insects:
            top_insect = max(insects, key=lambda x: x["count"])
            top_insect_name = top_insect["insect"]
            top_insect_ratio = round((top_insect["count"] / total) * 100)
        else:
            top_insect_name, top_insect_ratio = "정보 없음", 0
        
        if hourly:
            top_hour = int(hourly[0]["hour"])
            hour_range = f"{top_hour}시~{top_hour+2}시"
        else:
            hour_range = "정보 없음"
        
        return (
            f"{date} 기준 {farm_idx}번 농장의 해충 탐지 요약입니다.\n"
            f"오늘은 총 {total}마리의 해충이 탐지되었고, "
            f"{top_insect_name}가 가장 많은 비중({top_insect_ratio}%)을 차지했어요.\n"
            f"{top_zone}에서 가장 많이 탐지되었고, {hour_range} 사이에 활동량이 높았습니다.\n\n"
            "위 내용을 인사말은 제외하고, 농장주에게 보고하는 2~3문장의 친절한 요약으로 작성해주세요. 존댓말 구어체로 부탁드립니다."
        )
    
    def _build_monthly_stats_prompt(self, data: Dict[str, Any], month: str, farm_idx: int) -> str:
        """월간 통계 프롬프트 생성"""
        total = data.get("totalCount", 0)
        top_zone = data.get("topZone", "정보 없음")
        insects = data.get("insectDistribution", [])
        
        if insects:
            top_insect = max(insects, key=lambda x: x["count"])
            top_insect_name = top_insect["insect"]
            top_insect_ratio = round((top_insect["count"] / total) * 100)
        else:
            top_insect_name = "정보 없음"
            top_insect_ratio = 0
        
        # 시간대 통계는 details에서 계산
        details = data.get("details", [])
        hour_counter = Counter()
        for d in details:
            time_str = d.get("time")
            if time_str:
                try:
                    hour = int(time_str.split()[1].split(":")[0])
                    hour_counter[hour] += 1
                except:
                    pass
        
        if hour_counter:
            top_hour = hour_counter.most_common(1)[0][0]
            hour_range = f"{top_hour}시~{top_hour+2}시"
        else:
            hour_range = "정보 없음"
        
        return (
            f"{month} 동안 {farm_idx}번 농장의 해충 탐지 요약입니다.\n"
            f"총 {total}마리의 해충이 탐지되었고, {top_insect_name}가 가장 많은 비중({top_insect_ratio}%)을 차지했습니다.\n"
            f"가장 많이 탐지된 구역은 {top_zone}이며, {hour_range} 시간대에 해충 활동이 가장 활발했습니다.\n\n"
            "위 내용을 바탕으로 인사말 없이, 농장주에게 전달할 2~3문장의 요약을 존댓말 구어체로 작성해 주세요."
        )
    
    def _build_yearly_stats_prompt(self, data: Dict[str, Any], year: str, farm_idx: int) -> str:
        """연간 통계 프롬프트 생성"""
        total = data.get("totalCount", 0)
        top_zone = data.get("topZone", "정보 없음")
        insects = data.get("insectDistribution", [])
        
        if insects:
            top_insect = max(insects, key=lambda x: x["count"])
            top_insect_name = top_insect["insect"]
            top_insect_ratio = round((top_insect["count"] / total) * 100)
        else:
            top_insect_name = "정보 없음"
            top_insect_ratio = 0
        
        details = data.get("details", [])
        hour_counter = Counter()
        for d in details:
            time_str = d.get("time")
            if time_str:
                try:
                    hour = int(time_str.split()[1].split(":")[0])
                    hour_counter[hour] += 1
                except:
                    pass
        
        if hour_counter:
            top_hour = hour_counter.most_common(1)[0][0]
            hour_range = f"{top_hour}시~{top_hour+2}시"
        else:
            hour_range = "정보 없음"
        
        return (
            f"{year} 동안 {farm_idx}번 농장의 해충 탐지 요약입니다.\n"
            f"총 {total}마리의 해충이 탐지되었고, {top_insect_name}가 가장 많은 비중({top_insect_ratio}%)을 차지했습니다.\n"
            f"가장 많이 탐지된 구역은 {top_zone}이며, {hour_range} 시간대에 해충 활동이 가장 활발했습니다.\n\n"
            "위 내용을 바탕으로 인사말 없이, 농장주에게 전달할 2~3문장의 요약을 존댓말 구어체로 작성해 주세요."
        )