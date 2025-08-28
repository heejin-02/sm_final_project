"""
메타데이터 관리 서비스
탐지 결과를 CSV 파일로 관리
"""

import csv
import asyncio
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class MetadataService:
    """메타데이터 관리 서비스"""
    
    def __init__(self, base_dir: str = "data/metadata"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 메모리 캐시 (빠른 접근을 위해)
        self.rec_id_counter = {}  # camera_id별 녹화 ID 카운터
        self.track_id_counter = {}  # camera_id별 트래킹 ID 카운터
        
        # CSV 헤더 정의
        self.csv_headers = [
            "rec_id", "timestamp", "camera_id", "gh_idx", "label", 
            "confidence", "track_id", "x_min", "y_min", "x_max", "y_max", "file_path"
        ]
    
    def _get_csv_path(self, camera_id: str, target_date: date = None) -> Path:
        """CSV 파일 경로 반환"""
        if target_date is None:
            target_date = date.today()
        
        filename = f"{camera_id}_{target_date.strftime('%Y%m%d')}.csv"
        return self.base_dir / filename
    
    def _initialize_csv_file(self, csv_path: Path):
        """CSV 파일 초기화 (헤더 작성)"""
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.csv_headers)
            logger.info(f"새 메타데이터 파일 생성: {csv_path}")
    
    def _get_next_rec_id(self, camera_id: str) -> int:
        """다음 녹화 ID 반환"""
        if camera_id not in self.rec_id_counter:
            # 기존 파일에서 최대 rec_id 찾기
            csv_path = self._get_csv_path(camera_id)
            max_rec_id = 0
            
            if csv_path.exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            try:
                                rec_id = int(row['rec_id'])
                                max_rec_id = max(max_rec_id, rec_id)
                            except (ValueError, KeyError):
                                continue
                except Exception as e:
                    logger.error(f"CSV 파일 읽기 오류: {e}")
            
            self.rec_id_counter[camera_id] = max_rec_id
        
        self.rec_id_counter[camera_id] += 1
        return self.rec_id_counter[camera_id]
    
    def _get_next_track_id(self, camera_id: str) -> int:
        """다음 트래킹 ID 반환"""
        if camera_id not in self.track_id_counter:
            self.track_id_counter[camera_id] = 0
        
        self.track_id_counter[camera_id] += 1
        return self.track_id_counter[camera_id]
    
    async def save_detection_metadata(self, 
                                    camera_id: str,
                                    gh_idx: int,
                                    insect_name: str,
                                    confidence: float,
                                    bbox: List[int],
                                    timestamp: float = None,
                                    file_path: str = None) -> Dict:
        """
        탐지 메타데이터 저장
        
        Returns:
            저장된 메타데이터 딕셔너리
        """
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # CSV 파일 경로
            csv_path = self._get_csv_path(camera_id)
            self._initialize_csv_file(csv_path)
            
            # ID 생성
            rec_id = self._get_next_rec_id(camera_id)
            track_id = self._get_next_track_id(camera_id)
            
            # 메타데이터 구성
            metadata = {
                "rec_id": rec_id,
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H.%M%S.%f"),
                "camera_id": camera_id,
                "gh_idx": gh_idx,
                "label": insect_name,
                "confidence": confidence,
                "track_id": track_id,
                "x_min": bbox[0] / 1024,  # 정규화 (assuming 1024 width)
                "y_min": bbox[1] / 768,   # 정규화 (assuming 768 height)
                "x_max": bbox[2] / 1024,
                "y_max": bbox[3] / 768,
                "file_path": file_path or ""
            }
            
            # CSV에 데이터 추가
            with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writerow(metadata)
            
            logger.info(f"메타데이터 저장 완료: {camera_id} -> {insect_name} (REC:{rec_id}, TRACK:{track_id})")
            
            return metadata
            
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
            raise
    
    async def get_detection_history(self, 
                                  camera_id: str, 
                                  target_date: date = None,
                                  limit: int = 100) -> List[Dict]:
        """탐지 기록 조회"""
        try:
            csv_path = self._get_csv_path(camera_id, target_date)
            
            if not csv_path.exists():
                return []
            
            detections = []
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    detections.append(row)
            
            return list(reversed(detections))  # 최신순으로 정렬
            
        except Exception as e:
            logger.error(f"탐지 기록 조회 실패: {e}")
            return []
    
    async def get_daily_statistics(self, 
                                 camera_id: str, 
                                 target_date: date = None) -> Dict:
        """일일 통계 조회"""
        try:
            csv_path = self._get_csv_path(camera_id, target_date)
            
            if not csv_path.exists():
                return self._empty_stats()
            
            stats = {
                "total_detections": 0,
                "insect_counts": {},
                "avg_confidence": 0,
                "detection_times": [],
                "camera_id": camera_id,
                "date": target_date or date.today()
            }
            
            confidences = []
            
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    stats["total_detections"] += 1
                    
                    # 해충별 카운트
                    insect = row.get("label", "unknown")
                    stats["insect_counts"][insect] = stats["insect_counts"].get(insect, 0) + 1
                    
                    # 신뢰도
                    try:
                        conf = float(row.get("confidence", 0))
                        confidences.append(conf)
                    except ValueError:
                        pass
                    
                    # 탐지 시간
                    stats["detection_times"].append(row.get("timestamp", ""))
            
            # 평균 신뢰도 계산
            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)
            
            return stats
            
        except Exception as e:
            logger.error(f"일일 통계 조회 실패: {e}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict:
        """빈 통계 반환"""
        return {
            "total_detections": 0,
            "insect_counts": {},
            "avg_confidence": 0,
            "detection_times": [],
            "camera_id": "",
            "date": date.today()
        }
    
    async def cleanup_old_files(self, days_to_keep: int = 30):
        """오래된 메타데이터 파일 정리"""
        try:
            current_date = date.today()
            cutoff_date = current_date.replace(day=current_date.day - days_to_keep)
            
            deleted_count = 0
            for csv_file in self.base_dir.glob("*.csv"):
                try:
                    # 파일명에서 날짜 추출
                    parts = csv_file.stem.split('_')
                    if len(parts) >= 2:
                        date_str = parts[-1]  # YYYYMMDD
                        file_date = datetime.strptime(date_str, '%Y%m%d').date()
                        
                        if file_date < cutoff_date:
                            csv_file.unlink()
                            deleted_count += 1
                            logger.info(f"오래된 메타데이터 파일 삭제: {csv_file}")
                            
                except Exception as e:
                    logger.warning(f"파일 삭제 실패 {csv_file}: {e}")
            
            logger.info(f"메타데이터 정리 완료: {deleted_count}개 파일 삭제")
            
        except Exception as e:
            logger.error(f"메타데이터 정리 실패: {e}")
    
    def get_all_camera_ids(self) -> List[str]:
        """등록된 모든 카메라 ID 조회"""
        camera_ids = set()
        
        for csv_file in self.base_dir.glob("*.csv"):
            try:
                parts = csv_file.stem.split('_')
                if len(parts) >= 2:
                    camera_id = '_'.join(parts[:-1])  # 날짜 부분 제외
                    camera_ids.add(camera_id)
            except Exception:
                continue
        
        return sorted(list(camera_ids))