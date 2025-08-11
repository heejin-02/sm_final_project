# 벌레잡는109 - YOLOv5 탐지 실행 가이드

이 저장소는 실시간 해충 탐지를 위한 YOLOv5 기반 탐지 코드를 제공합니다.  
본 파일은 영상 촬영, 업로드, 분석 결과 전송 및 FastAPI 서버와 연동된 전화·GPT 요약 요청까지 자동으로 처리됩니다.

---

## ✅ 이 코드는 누구를 위한 것인가요?

- FastAPI 서버는 **운영자가 실행**합니다.
- 이 코드는 **탐지 전용 클라이언트용**입니다.
- 클라이언트는 FastAPI 서버와 Spring 서버에 자동으로 요청을 보냅니다.

---

## 🧩 구성 파일

- `detect_only.py`: YOLOv5 탐지 및 영상 전송 실행 파일
- `.env`: 전화 발신용 SignalWire 환경변수 (전화 미사용 시 없어도 됨)

---

## ⚙️ 실행 전 준비사항

1. **YOLOv5 모델 가중치** 준비  
   → `best_clean.pt` 또는 본인의 모델 파일

2. **웹캠이 연결되어 있어야 합니다**

3. `.env` 파일 예시 (전화 알림이 필요한 경우)
   ```env
   # OPENAI_API_KEY 설정
   OPENAI_API_KEY = sk-...

   # Twilio
   # TWILIO_ACCOUNT_SID = ...
   # TWILIO_AUTH_TOKEN = ...
   # TWILIO_PHONE_NUMBER = ...
   # USER_PHONE_NUMBER = ...

   # DB 연결
   DB_USER = "joo"
   DB_PASS = "smhrd4"
   DB_DSN = "project-db-campus.smhrd.com:1523/xe"

   # SignalWire
   SIGNALWIRE_PROJECT_ID=...
   SIGNALWIRE_AUTH_TOKEN=...
   SIGNALWIRE_PHONE_NUMBER=...
   SIGNALWIRE_SPACE_URL=catchbug.signalwire.com
   ```

4. **`gh_idx` 설정 필수**
   ```python
   # detect_only.py 내부
   gh_idx = 24  # 각 사용자 고유값으로 수정
   ```

---

## 🚀 실행 방법

```bash
python detect_only.py --weights best_clean.pt --source 0 --view-img
```

- `--weights`: YOLOv5 가중치 파일
- `--source`: 웹캠 사용 시 `0`, 동영상 파일도 가능
- `--view-img`: 탐지 결과를 실시간으로 화면에 표시

---

## 🔗 서버 연동 정보

- 영상 업로드: `http://192.168.219.72:8095/api/qc-videos`
- 분석 결과 저장: `http://192.168.219.72:8095/api/qc-classification`
- 전화번호 조회: `http://192.168.219.72:8000/api/get-phone`
- GPT 요약 요청: `http://192.168.219.72:8000/api/summary-by-imgidx`

> 모든 요청은 운영자의 FastAPI 및 Spring 서버에 전송됩니다.

---

## 📞 전화 알림 및 요약 생성

- 벌레가 탐지되면 자동으로:
  - 영상 업로드
  - 분석 결과 저장
  - 사용자에게 전화 발신
  - GPT 요약 요청

> 전화/요약은 FastAPI 서버의 `/twilio-call`, `/summary-by-imgidx` 경로를 사용합니다.

---

## ❓ 문제 발생 시

- `gh_idx` 값이 잘못되었을 수 있음 → FastAPI 서버에 등록된 사용자 확인
- FastAPI 서버가 꺼져있으면 전화, 요약 호출 실패
- SignalWire 설정 오류 시 전화 발신 실패

---

## 🙋‍♀️ 문의

운영자에게 문의하세요: **오희진**