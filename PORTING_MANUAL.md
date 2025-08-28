# 벌레잡는 109 - 포팅 매뉴얼

## 개요
실시간 해충 탐지 및 음성 알림 서비스를 다른 네트워크 환경에서 배포하기 위한 가이드입니다.

## 시스템 구성
- **Backend (Spring Boot)**: 포트 8095
- **Frontend (React)**: 포트 5173 (개발) / 정적 배포 (운영)
- **ML API (Python FastAPI)**: 포트 8003

## 서버 시작 순서

### 1. 데이터베이스 확인
Oracle DB 서버가 실행 중인지 확인
```
- 호스트: project-db-campus.smhrd.com:1523:xe
- 사용자: joo / smhrd4
```

### 2. Backend (Spring Boot) 시작
```bash
cd backend
./mvnw spring-boot:run
```
**시작 확인**: http://localhost:8095/api/health 또는 Swagger UI

### 3. ML API 서버 시작
```bash
cd ml
source venv/bin/activate  # 가상환경 활성화
python unified_api.py
```
**시작 확인**: http://localhost:8003/docs (FastAPI 문서)

### 4. Frontend 개발 서버 시작 (개발용)
```bash
cd front
npm install
npm run dev
```
**접속**: http://localhost:5173

## IP 설정 변경 위치

### 🔧 네트워크 환경 변경 시 수정해야 할 파일들

#### 1. Frontend 설정
**파일**: `/front/.env`
```env
VITE_API_BASE_URL=http://192.168.219.47:8095
```
- WiFi IP 주소로 변경
- 형식: `http://[서버_IP_주소]:8095`

#### 2. Backend 설정  
**파일**: `/backend/src/main/resources/application.properties`
```properties
server.port=8095
server.address=0.0.0.0  # 모든 네트워크 인터페이스에서 접근 허용

# ML API 서버 주소
ml.api.base-url=http://localhost:8003  # 동일 서버인 경우 localhost 유지
```

#### 3. Backend 비디오 URL 생성 (자동 감지)
**파일**: `/backend/src/main/java/com/smhrd/web/QcImage/QcImageController.java:64-66`
```java
String serverIp = java.net.InetAddress.getLocalHost().getHostAddress(); // 자동 감지
String serverPort = "8095";
String videoUrl = "http://" + serverIp + ":" + serverPort + "/videos/" + dateFolder + "/" + fileName;
```
- **동작방식**: 서버 IP를 자동으로 감지하여 비디오 URL 생성
- **수동 설정 필요한 경우**: `serverIp` 변수를 고정 IP로 설정

#### 4. ML API 설정 (필요 시)
**파일**: `/ml/unified_api.py`
```python
# Spring Boot API 서버 주소
SPRING_BOOT_BASE_URL = "http://localhost:8095"  # 다른 서버인 경우 IP 변경
```

## WiFi 네트워크 배포 체크리스트

### 📋 배포 전 확인사항
1. **서버 IP 주소 확인**
   ```bash
   # Windows
   ipconfig
   
   # Linux/Mac
   ifconfig 또는 ip addr show
   ```

2. **파일 수정**
   - [ ] `/front/.env` - VITE_API_BASE_URL 업데이트
   - [ ] 필요시 `/ml/unified_api.py` - SPRING_BOOT_BASE_URL 업데이트
   - [ ] Backend는 자동 IP 감지 사용 (수정 불필요)

3. **방화벽 설정**
   - [ ] 포트 8095 (Backend) 열기
   - [ ] 포트 5173 (Frontend 개발용) 열기  
   - [ ] 포트 8003 (ML API) 열기

4. **서비스 재시작**
   ```bash
   # 1. Backend 재시작
   cd backend && ./mvnw spring-boot:run
   
   # 2. ML API 재시작
   cd ml && python unified_api.py
   
   # 3. Frontend 재시작
   cd front && npm run dev
   ```

### 🌐 접속 URL 예시
- **Frontend**: http://192.168.219.47:5173
- **Backend API**: http://192.168.219.47:8095
- **ML API**: http://192.168.219.47:8003

## 주요 기능 테스트

### 1. 웹캠 해충 탐지
```bash
cd ml/model/yolov5
python detect_webcam.py
```

### 2. 비디오 업로드/탐지
1. 프론트엔드에서 비디오 업로드
2. Backend에서 ML API 호출
3. 탐지 결과 및 GPT 분석 확인

### 3. 전화 알림 (SignalWire)
- 해충 탐지 시 자동 전화 발송
- 테스트용 전화번호: 설정 파일에서 확인

## 문제 해결

### 자주 발생하는 오류
1. **CORS 에러**: Backend WebConfig.java의 CORS 설정 확인
2. **비디오 로드 실패**: 비디오 파일 경로 및 Static Resource 설정 확인
3. **API 통신 실패**: IP 주소 및 포트 번호 재확인

### 로그 확인
- **Backend**: Spring Boot 콘솔 로그
- **Frontend**: 브라우저 개발자 도구 Console
- **ML API**: Python 콘솔 로그

## 운영 배포 (선택사항)

### Frontend 프로덕션 빌드
```bash
cd front
npm run build
# dist 폴더를 웹서버(Nginx/Apache)에 배포
```

### Backend 프로덕션 실행
```bash
cd backend
./mvnw clean package
java -jar target/FinalProject-0.0.1-SNAPSHOT.jar
```

---
**작성일**: 2025-08-28  
**버전**: v1.0