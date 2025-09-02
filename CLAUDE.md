# CLAUDE.md - AI Assistant Instructions for "벌레잡는 109" Project

## 프로젝트 개요
이 프로젝트는 **실시간 해충 탐지 및 음성 알림 서비스**입니다. 고령 농업인을 위한 웹캠 기반 해충 탐지 + 전화 알림 자동화 솔루션으로, YOLOv5 모델을 사용하여 토마토 주요 병해충을 탐지하고 즉시 음성전화로 농장주에게 알립니다.

## 프로젝트 구조

### Backend (Spring Boot)
- **위치**: `/backend`
- **기술 스택**: Spring Boot 3.5.3, Java 17, Oracle DB, MyBatis
- **주요 기능**:
  - 사용자 인증 및 관리
  - 농장 정보 관리
  - 해충 탐지 데이터 처리
  - 리포트 생성 및 통계
  - Swagger API 문서화
  - 웹소켓 기반 실시간 통신
  - 비디오 스트리밍 서비스

### Frontend (React)
- **위치**: `/front`
- **기술 스택**: React 19, Vite, Tailwind CSS, Recharts
- **주요 기능**:
  - 실시간 해충 탐지 대시보드
  - 농장 모니터링 UI
  - 통계 시각화
  - 관리자 페이지
  - 알림 관리
  - 반응형 디자인
  - 데이터 캐싱 최적화

### ML/AI (Python)
- **위치**: `/ml`
- **기술 스택**: YOLOv5, FastAPI, LangChain, ChromaDB, SignalWire
- **주요 기능**:
  - 실시간 해충 탐지 (YOLOv5)
  - 웹캠 영상 처리 및 비디오 저장
  - 방제 챗봇 (RAG 구조)
  - GPT 기반 리포트 요약
  - SignalWire 기반 음성 전화 알림 서비스
  - 메타데이터 관리 및 CSV 저장
  - 탐지 이미지 크롭 및 저장

## 개발 명령어

### Backend
```bash
cd backend
./mvnw spring-boot:run  # 서버 실행
./mvnw clean install     # 빌드
```

### Frontend
```bash
cd front
npm install              # 의존성 설치
npm run dev              # 개발 서버 실행
npm run build            # 프로덕션 빌드
npm run type-check       # TypeScript 체크
```

### ML/AI
```bash
cd ml
pip install -r requirements.txt               # 전체 의존성 설치
python app/main.py                           # 통합 API 서버 실행 (포트 8003)
python model/yolov5/detect.py                # 해충 탐지 실행 (테스트용)
python model/yolov5/detect_webcam.py         # 웹캠 실시간 탐지
```

## 주요 API 엔드포인트

### 사용자 관리
- `POST /api/users/login` - 로그인
- `GET /api/users/info` - 사용자 정보 조회
- `POST /api/admin/insert-user` - 사용자 추가 (관리자)

### 농장 관리
- `GET /api/farm/details` - 농장 상세 정보
- `POST /api/farm/feedback` - 피드백 제출
- `GET /api/alert` - 알림 목록 조회
- `GET /api/alert/detail/{alertId}` - 알림 상세 정보
- `GET /api/today` - 오늘의 탐지 현황

### ML 연동 API
- `POST /api/ml/detect` - ML 해충 탐지 요청 (프록시)
- `POST /api/ml/upload` - 비디오 업로드 및 처리
- `GET /api/video/stream/{filename}` - 비디오 스트리밍

### ML API (FastAPI - 포트 8003)
- `POST /api/upload` - 비디오 파일 업로드 및 처리
- `POST /api/chatbot` - RAG 기반 방제 챗봇
- `GET /api/gpt/{alert_id}` - GPT 요약 생성
- `WS /api/ws/camera/{camera_id}` - 웹소켓 실시간 영상 처리

### SignalWire 전화 알림 (ML API - 포트 8003)
- `POST /api/make-call` - SignalWire 전화 발신
- `GET /api/get-phone` - 온실 인덱스로 전화번호 조회
- `GET /api/signalwire/voice` - TwiML 음성 메시지 생성
- `GET /api/call-history` - 통화 기록 조회

### 리포트 및 통계
- `GET /api/report/daily` - 일일 통계
- `GET /api/report/monthly` - 월간 통계
- `GET /api/report/yearly` - 연간 통계
- `GET /api/report/detection-details` - 탐지 상세 정보
- `GET /api/report/insect-distribution` - 해충 분포도
- `GET /api/report/zone-counts` - 구역별 탐지 수

## 데이터베이스 구조
- **사용자 테이블**: 사용자 정보 및 권한
- **농장 테이블**: 농장 정보 및 온실 데이터
- **탐지 테이블**: 해충 탐지 기록
- **알림 테이블**: 알림 발송 기록
- **피드백 테이블**: 사용자 피드백

## 해충 분류 인덱스
```python
# 현재 탐지 가능한 해충 (10종)
INSECT_INDEX = {
    "꽃노랑총채벌레": 1,
    "담배가루이": 2,
    "비단노린재": 3,
    "알락수염노린재": 4,
    "온실가루이": 5,
    "차먼지응애": 6,
    "점박이응애": 7,
    "아메리카잎굴파리": 8,
    "파밤나방": 9,
    "담배거세미나방": 10
}

# YOLOv5 모델 클래스 인덱스
class_names = ['꽃노랑총채벌레', '담배가루이', '비단노린재', '알락수염노린재']
```

## 환경 변수 설정

### Backend (application.properties)
```properties
# 서버 설정
server.port=8095
server.address=0.0.0.0

# 데이터베이스 설정
spring.datasource.url=jdbc:oracle:thin:@project-db-campus.smhrd.com:1523:xe
spring.datasource.username=joo
spring.datasource.password=smhrd4

# MyBatis 설정
mybatis.configuration.map-underscore-to-camel-case=true
mybatis.mapper-locations=classpath:mapper/*.xml

# ML API 설정
ml.api.base-url=http://localhost:8003
ml.api.timeout=10000

# 비디오 저장 설정
file.upload.dir = ./videos
spring.servlet.multipart.max-file-size=100MB
spring.servlet.multipart.max-request-size=100MB

# 웹소켓 설정
spring.websocket.allowed-origins=http://localhost:5173,http://192.168.219.49:5173

# CORS 설정
cors.allowed-origins=http://localhost:5173,http://192.168.219.49:5173
```

### Frontend (.env)
```properties
VITE_API_BASE_URL=http://192.168.219.49:8095
```

### ML/AI (.env)
```properties
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# SignalWire 설정
SIGNALWIRE_PROJECT_ID=your_project_id
SIGNALWIRE_AUTH_TOKEN=your_auth_token
SIGNALWIRE_PHONE_NUMBER=+1234567890
SIGNALWIRE_SPACE_URL=your_space.signalwire.com

# Spring Boot 연동
SPRING_BOOT_URL=http://192.168.219.49:8095
FASTAPI_PORT=8003
FASTAPI_HOST=0.0.0.0
```

## 코드 스타일 가이드
- **Java**: Spring Boot 표준 컨벤션
- **JavaScript/React**: ESLint 규칙 준수
- **Python**: PEP 8 스타일 가이드

## 주의사항
1. **보안**: 절대로 API 키나 비밀번호를 커밋하지 마세요
2. **테스트**: 코드 변경 시 관련 테스트 실행 확인
3. **문서화**: API 변경 시 Swagger 문서 업데이트
4. **커밋**: 의미 있는 커밋 메시지 작성

## 팀 협업 규칙
- 기능별 브랜치 생성 후 main 브랜치에 머지
- PR 생성 시 코드 리뷰 필수
- 이슈 트래킹을 통한 작업 관리

## 문제 해결
- **CORS 에러**: WebConfig.java에서 CORS 설정 확인 (현재 192.168.219.* 대역 허용)
- **DB 연결 실패**: application.properties의 DB 설정 확인
- **YOLOv5 모델 로드 실패**: 
  - weights 파일 경로 확인 (best.pt, best_clean.pt)
  - GPU 메모리 부족: batch_size 조정
  - CUDA 버전 호환성 확인
- **ML API 연결 실패**: 
  - ML 서버(포트 8003) 실행 상태 확인
  - 방화벽 설정 확인
  - RestTemplate timeout 설정 확인
- **비디오 스트리밍 오류**: 
  - videos 디렉토리 권한 및 경로 확인
  - FFmpeg 설치 확인
  - 비디오 코덱 호환성 (H.264 권장)
- **웹소켓 연결 실패**:
  - WebSocketConfig 설정 확인
  - 포트 및 엔드포인트 경로 확인
  - 클라이언트-서버 프로토콜 일치 확인
- **네트워크 접속 문제**: 
  - 같은 와이파이 환경에서 IP 주소 확인 (192.168.219.49)
  - 방화벽 포트 개방 (8095, 8003, 5173)
- **SignalWire 전화 발신 실패**: 
  - 401 Unauthorized: SignalWire 인증 정보 확인 (.env 파일)
  - 국제발신 오류: SignalWire 계정에서 국제발신 승인 필요
  - 전화번호 형식: +82 형식으로 정규화 확인
- **전화번호 조회 실패**: MyBatis 결과 매핑 대소문자 확인 (USERPHONE vs userPhone)
- **메모리 부족 문제**:
  - JVM 힙 메모리 증가: -Xmx4g
  - 비디오 처리 시 임시 파일 정리
  - 캐시 설정 최적화

## 프로젝트 파일 구조
- `general_index.md`: 프로젝트 개요 및 핵심 가치
- `detailed_index.md`: 상세 기술 문서 및 API 명세
- `PORTING_MANUAL.md`: 설치 및 배포 가이드
- `README.md`: 프로젝트 소개

## 연락처
- 백엔드 담당: 주연진
- ML/AI 담당: 오희진  
- 프론트엔드 담당: 한수연