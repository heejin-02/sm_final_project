# [2025] 벌레잡는 109(백구) 🐛🐕 <br/>
**10초 만에 탐지해서 9초 안에 알려준다!**  
고령 농업인을 위한 **실시간 해충 탐지 & 음성 알림 서비스**
<br/>
<br/>

### 🛡️ 프로젝트 개요
**벌레잡는 109**는 **토마토 주요 병해충을 실시간으로 탐지**하고,  
탐지 즉시 **전화 음성 알림**을 발송하는 AI 기반 방제 보조 서비스입니다.  
고령 농업인을 위한 직관적인 **전화 알림**과, **웹 기반 대시보드**를 통해  
탐지 이력 및 발생률을 시각화하여 손쉽게 확인할 수 있습니다.  

### 🧠 기획 배경
- **농업 현장 문제**  
  - 해충은 농작물 생산량에 직접적인 피해를 유발  
  - 고령 농업인들은 작은 벌레 탐지가 어렵고, 스마트폰 앱 사용도 불편  
- **현실적 필요**  
  - 신속한 탐지 + 직관적인 알림 수단 필요  
  - 문자/앱 알림 대신 **전화 알림**을 통한 즉각적인 대응 지원  
- **목표**  
  - 10초 내 탐지, 9초 내 전화 알림 → 농업인의 신속한 방제 활동 지원  

### 🛠 기술 스택
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/WebSocket-000000?style=flat-square&logo=socket.io&logoColor=white"/>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/SignalWire-1D8FE1?style=flat-square&logo=twilio&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0E5A89?style=flat-square"/>
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=flat-square"/>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/TailwindCSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white"/>
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/SpringBoot-6DB33F?style=flat-square&logo=springboot&logoColor=white"/>
  <img src="https://img.shields.io/badge/OracleDB-F80000?style=flat-square&logo=oracle&logoColor=white"/>
</p>


### 🖼 화면 구성
| #메인 페이지 | #큰글씨 모드 |
|:---:|:---:|
| ![메인페이지 (압축)](https://github.com/user-attachments/assets/7a4c7557-ef05-4962-a3a5-8f6b541f1dfe) | ![사이트 크게작게 보기](https://github.com/user-attachments/assets/4c89e811-b7b5-4a7e-bdcd-8d8fb67c91ea) | 
| #농장상세 페이지 | #농장알림 상세 페이지 |
| ![농장 상세 페이지 (압축)](https://github.com/user-attachments/assets/074db3d8-58b7-4c5f-bda2-3fbd7a96a551) | ![농장 알림 상세 페이지](https://github.com/user-attachments/assets/35ebdaed-a662-497d-a970-e3e961d7f46c) |
| #일간통계 | #월간통계 |
| ![일간통계](https://github.com/user-attachments/assets/62b02610-c017-48a9-a731-6dbd95617a5e) | ![월간통계](https://github.com/user-attachments/assets/e9779bf5-506b-45d8-93e3-0ffd60597c00) | 
| #연간통계 | (__) |
| ![연간통계](https://github.com/user-attachments/assets/0062fbfc-ae99-484a-a2a0-f17eea6a3c7c) | (__) |
| #관리자 페이지 | #회원등록 및 농장등록 | 
| ![관리자페이지](https://github.com/user-attachments/assets/67982c48-eea3-41bf-a32f-b08407c36405) | ![회원 및 농장 등록](https://github.com/user-attachments/assets/873a84f8-4694-4116-8ed1-3c710fcd2520) | 


### 🚀 주요 기능
- 🐛 **실시간 해충 탐지 (YOLOv5 + OpenCV)**  
- 🎥 탐지 시 **10초간 영상 자동 저장**  
- 📞 **전화 음성 알림 발송 (SignalWire API)**  
- 📊 **탐지 이력/발생률 통계** 제공 (Chart.js 시각화)  
- 🧠 **해충 맞춤 방제 챗봇** (LangChain + ChromaDB 기반)  
- 🧑‍🌾 **고령자 친화형 UI** (큰 글씨, 다크모드, 음성 버튼 지원)  



### 💻 서비스 아키텍처
---
Webcam에서 촬영된 영상은 YOLOv5 감지 서버로 전송되며, 서버에서는 10초 단위로 영상을 저장합니다. <br/> 감지 서버는 동시에 여러 작업을 수행하는데, WebSocket을 통해 프론트엔드에 실시간 전화알림을 보내고, SignalWire TTS를 이용해 전화로 음성 알림을 제공합니다. 또한 GPT API를 활용하여 벌레 리포트를 생성하며, LangChain과 ChromaDB를 활용해 방제 관련 챗봇 응답도 제공합니다.

<img width="700" src="https://github.com/user-attachments/assets/128ec166-5549-4ddc-b1ac-735537dd5698" />
<br/>

### 💣 백엔드 트러블 슈팅
---
 ### 1. 이미지 저장 및 경로 설정 이슈
 **발생 배경🚨**  
이미지를 DB에 직접 저장하는 대신 문자열(Base64) 형태로 저장하고, 실제 파일은 로컬 디렉토리에 저장하는 방식을 적용하고자 함. 그러나, 저장 경로 지정 과정에서 프로젝트 내부와 로컬 디렉토리 간 경로 매핑이 필요.  

**문제 원인😅**  
- 리소스 폴더(`/resources`) 내 이미지 디렉토리와 로컬 저장 경로(`C:` 드라이브) 간의 일관성 있는 경로 설정 부재  
- MultipartFile 업로드 시 MediaType 설정 누락으로 파일 업로드 처리 오류 발생  

**해결 방안💡**  
- `/resources/images` 디렉토리를 생성하여 내부 저장소 경로를 지정  
- 로컬 저장소(`C:/images`)에도 동일한 구조를 생성해 운영 환경과 로컬 환경의 경로 동기화  
- Spring Controller에서 `MultipartFile` 타입으로 수신하고, `MediaType`을 명시적으로 설정하여 이미지 처리 호환성 확보

### 2. 유저 기반 농장 리스트 조회 시 `null` 값 반환
**발생 배경🚨**  
Swagger에서 특정 유저 정보로 농장 리스트 조회 시 모든 필드가 `null`로 반환되는 문제 발생.

**문제 원인😅**  
- Service/Mapper 계층에서 조회 로직은 정상이나, Controller에서 DTO 매핑 시 잘못된 객체를 사용.
- 실제 데이터는 `UserDTO`에서 가져와야 하는데, `FarmDetailDTO` 객체에 값을 주입하려 시도.
- `FarmDetailDTO`에는 해당 데이터가 존재하지 않아 결과적으로 `null` 반환.

**해결 방안💡**  
- Controller 로직 수정: `UserDTO`에서 사용자 식별값을 추출하고, 이를 기반으로 농장 조회 쿼리 실행.
- DTO 매핑 구조 재점검 및 역할 분리, 불필요한 DTO 혼용 방지.

### 3. XML 문법 특수문자 처리 오류

**발생 배경🚨**  
MyBatis Mapper XML 작성 시 `<`, `>`, `<=` 등의 연산자 사용으로 XML 파싱 오류 발생.

**문제 원인😅**  
- XML 문서에서 `<`, `>` 등 특수문자는 예약어로 인식되어 escape 처리 필요.
- 예: `LEVEL <= 12` 구문 사용 시 XML 파서에서 문법 오류 발생.

**해결 방안💡**  
- MyBatis Mapper XML에서 연산자 escape 처리:
    ```xml
    복사편집
    <=  →  &lt;=
    >=  →  &gt;=
    ```
- 수정 쿼리 : 
    ```xml
    FROM (
        SELECT LEVEL AS MONTH
        FROM dual
        CONNECT BY LEVEL &lt;= 12
    )
    ```
- XML 문법 규칙 준수 및 향후 쿼리 작성 시 특수문자 처리 가이드 마련.

### 4. 연별·계절별 통계 쿼리 작성 오류

**발생 배경🚨**
연도별 해충 통계 및 계절별 예측 기능 구현 과정에서 일부 쿼리가 정상적으로 데이터 집계를 수행하지 못함. 특히 예측 로직(`predicted2026`) 계산에서 값이 누락되거나 비정상적으로 크게 계산되는 현상이 발생.

**문제 원인😅**  
1. **CASE 절에서 ELSE 0 미포함**
    - 안되던 쿼리:
        ```sql
        SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2024' THEN 1 END)
        ```
        - 조건 불만족 시 **NULL**이 반환되고, `SUM`은 NULL을 무시하므로 총합이 의도치 않게 작게 계산됨.
        - 예: 10개 중 5개가 조건 불만족이면 나머지 5개는 `NULL` → 합계에서 제외됨.
    - 수정된 쿼리:
        ```sql
        SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2024' THEN 1 ELSE 0 END)
        ```
        - 조건 불만족 시 **0**을 넣어 안전하게 합산 가능.
        - 집계 로직 안정성 확보.
2. **문자열 비교 시 공백 문제**
    - 안되던 쿼리:
        ```sql
        C.ANLS_RESULT = I.INSECT_NAME
        ```
        - DB에 저장된 해충명(`ANLS_RESULT`)이나 기준 데이터(`INSECT_NAME`)에 **앞뒤 공백**이 포함되면 매칭 실패.
    - 수정된 쿼리:
        ```sql
        TRIM(C.ANLS_RESULT) = TRIM(I.INSECT_NAME)
        ```
        - 불필요한 공백 제거 후 비교 → 매칭률 향상.
        - 특히 문자열 데이터가 수동 입력되거나 다른 시스템에서 이관된 경우 공백 문제 빈번.
3. **연도 필터링 미흡**
    - 안되던 쿼리:
        -- 연도 제한 조건이 없음
        - 모든 연도의 데이터가 집계에 포함되어 예측치가 왜곡됨.
        - 2024·2025년 외 데이터가 포함되면, 다음 해 예측(`predicted2026`) 계산이 부정확해짐.
    - 수정된 쿼리:
        ```sql
        AND TO_CHAR(C.CREATED_AT, 'YYYY') IN ('2024', '2025')
        ```
        - 분석 대상 연도만 한정하여 집계 → 예측 계산의 정확도 향상.

- 그렇게 애먹인 코드의 완성본
    ```sql
    SELECT 
        CASE 
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('03', '04', '05') THEN 'Spring'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('06', '07', '08') THEN 'Summer'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('09', '10', '11') THEN 'Fall'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('12', '01', '02') THEN 'Winter'
        END AS season,
        I.INSECT_NAME AS insectName,
        NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2024' THEN 1 ELSE 0 END), 0) AS count2024,
        NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2025' THEN 1 ELSE 0 END), 0) AS count2025,
        ROUND(
            CASE
                WHEN NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2024' THEN 1 ELSE 0 END), 0) = 0 THEN
                    NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2025' THEN 1 ELSE 0 END), 0)
                ELSE
                    NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2025' THEN 1 ELSE 0 END), 0) * 
                    NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2025' THEN 1 ELSE 0 END), 0) /
                    NVL(SUM(CASE WHEN TO_CHAR(C.CREATED_AT, 'YYYY') = '2024' THEN 1 ELSE 0 END), 0)
            END
        ) AS predicted2026
    FROM 
        QC_CLASSIFICATION C
    JOIN 
        QC_INSECT I ON TRIM(C.ANLS_RESULT) = TRIM(I.INSECT_NAME)
    WHERE 
        I.INSECT_NAME IN ('꽃노랑총채벌레', '담배가루이', '비단노린재', '알락수염노린재')
        AND TO_CHAR(C.CREATED_AT, 'YYYY') IN ('2024', '2025') 
    GROUP BY 
        CASE 
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('03', '04', '05') THEN 'Spring'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('06', '07', '08') THEN 'Summer'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('09', '10', '11') THEN 'Fall'
            WHEN TO_CHAR(C.CREATED_AT, 'MM') IN ('12', '01', '02') THEN 'Winter'
        END,
        I.INSECT_NAME
    ORDER BY 
        season, insectName;
    ```
**개선 효과**
- **NULL 값 방지**: `CASE ELSE 0`로 모든 조건에서 값 반환 보장.
- **매칭률 향상**: `TRIM()`으로 불필요한 공백 제거.
- **정확한 집계 범위**: 연도 필터로 분석 데이터 범위를 명확히 제한.
- **예측 정확도 개선**: 잘못된 데이터 포함 방지로 예측치 신뢰성 확보.

<br/>

### 📌 추후 개선사항
---
1. 기획 단계 고도화
- 단순 결과 중심이 아닌, 기획 과정에서의 의사결정 근거와 프로젝트 초반에 비즈니스 모델 캔버스나 수익성 분석 도구를 도입해 기획 단계에서부터 확장 가능성을 고려해볼 상황
- 수익성 및 비즈니스 모델 검증 고려, 기술 구현 외에도 비즈니스 관점에서의 KPI(예: 사용자 수, 전환율, 유지율)를 설정해 지속적으로 검증

2. 협업 방식 개선
- Swagger 기반 API 문서화를 더욱 체계화 → 자동화된 API 테스트(예: Postman, Newman)와 연계.
- 프론트엔드–백엔드 협업 시 버전 관리 규칙과 명세 변경 프로세스를 사전에 정의해 혼선을 줄임.

3. 기술적 한계 보완
- 노트북 내장 카메라 → 고성능 외부 장비 적용을 고려
- 장비 교체 전, **데이터 전처리 기법(해상도 보정, 노이즈 제거)**을 통해 정확도 개선을 시도.
- 추후 클라우드 리소스 활용(GPU 서버 등)으로 모델 학습 및 탐지 성능 향상.

4. 지속적인 성과 관리
- 프로젝트 완료 후에도 **회고(리트로스펙티브)**를 정례화하여 개선점을 팀 차원에서 축적.
- 개인적 성장뿐만 아니라, 조직 차원의 노하우로 전환해 다음 프로젝트에 재활용 가능하게 시스템화되도록 구축해보기

### 팀원 소개
| 주연진 <br/> 팀장/DB/백엔드 개발 | 오희진 <br/> ML/AI모델링/FAST API | 한수연 <br/> 기획/프론트엔드 | 
|:--:|:--:|:--:|
| __ | ++ | __ |
| RESTful API 개발, Web 설계 및 비즈니스 로직, 게시판 CRUD기능 구현, 페이징/검색 처리 | 해충데이터 전처리, YOLOv5학습 및 탐지 구현, 전화API 연동 및 기능 구현, RAG시스템 구축, GPT API통합 | 서비스 화면 기획, 고령자 대상 UI/UX 설계, 서비스 화면구현, 데이터 시각화 및 API연동 | 










