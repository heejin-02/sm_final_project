# ML 마이크로서비스 아키텍처

기존 단일 FastAPI 앱을 4개의 독립적인 마이크로서비스로 분리했습니다.

## 🏗️ 서비스 구조

### 1. RAG 서비스 (포트 8003)
- **기능**: 해충 방제 챗봇, 질의응답
- **의존성**: ChromaDB, LangChain, OpenAI Embeddings
- **엔드포인트**:
  - `/api/ask` - 일반 질문
  - `/api/chat` - 해충 컨텍스트 대화
  - `/api/summary-by-imgidx` - 이미지 기반 요약

### 2. GPT 요약 서비스 (포트 8004)
- **기능**: 통계 기반 요약 생성
- **의존성**: OpenAI API만 사용 (가장 가벼움)
- **엔드포인트**:
  - `/api/daily-gpt-summary` - 일간 요약
  - `/api/monthly-gpt-summary` - 월간 요약
  - `/api/yearly-gpt-summary` - 연간 요약

### 3. 전화 알림 서비스 (포트 8005)
- **기능**: SignalWire 음성 전화 알림
- **의존성**: SignalWire API
- **엔드포인트**:
  - `/api/get-phone` - 전화번호 조회
  - `/api/make-call` - 전화 발신
  - `/api/call-history` - 통화 기록

### 4. 파일 프록시 서비스 (포트 8006)
- **기능**: Spring Boot로 파일 전달
- **의존성**: 최소 (requests만 사용)
- **엔드포인트**:
  - `/api/upload` - 비디오 업로드

## 🚀 빠른 시작

### 1. 초기 설정
```bash
# 실행 권한 부여
chmod +x *.sh

# 모든 서비스 설정 (venv 생성 및 패키지 설치)
./setup_services.sh
```

### 2. 환경 변수 설정
`.env` 파일을 편집하여 API 키 입력:
```env
OPENAI_API_KEY=your_actual_key_here
SPRING_BOOT_URL=http://localhost:8095
# SignalWire 설정...
```

### 3. 서비스 실행

#### 개별 실행 (개발 시 권장)
```bash
./start_rag.sh     # RAG 서비스만
./start_gpt.sh     # GPT 서비스만
./start_phone.sh   # 전화 서비스만
./start_proxy.sh   # 프록시 서비스만
```

#### 전체 실행
```bash
./start_all.sh     # 모든 서비스 시작
./stop_all.sh      # 모든 서비스 중지
```

## 💡 장점

1. **빠른 시작**: 필요한 서비스만 실행 가능
2. **메모리 효율**: RAG 서비스(무거움)를 사용하지 않을 때 메모리 절약
3. **독립적 개발**: 각 서비스를 독립적으로 수정/재시작
4. **장애 격리**: 한 서비스 장애가 다른 서비스에 영향 없음

## 📝 개발 팁

- **RAG 개발 시**: `./start_rag.sh`만 실행
- **GPT 요약 테스트**: `./start_gpt.sh`만 실행
- **전체 통합 테스트**: `./start_all.sh` 실행

## 🔧 트러블슈팅

### 포트 충돌
```bash
# 특정 포트 사용 프로세스 확인
lsof -i:8003

# 강제 종료
kill -9 [PID]
```

### ChromaDB 초기화 오류
```bash
# ChromaDB 재설정
rm -rf chroma_db/
# 서비스 재시작
```

### 메모리 부족
- RAG 서비스만 중지: `kill $(lsof -ti:8003)`
- 나머지 서비스는 계속 실행 가능