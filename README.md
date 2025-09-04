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



### 서비스 아키텍처
---
Webcam에서 촬영된 영상은 YOLOv5 감지 서버로 전송되며, 서버에서는 10초 단위로 영상을 저장합니다. <br/> 감지 서버는 동시에 여러 작업을 수행하는데, WebSocket을 통해 프론트엔드에 실시간 전화알림을 보내고, SignalWire TTS를 이용해 전화로 음성 알림을 제공합니다. 또한 GPT API를 활용하여 벌레 리포트를 생성하며, LangChain과 ChromaDB를 활용해 방제 관련 챗봇 응답도 제공합니다.

<img width="700" src="https://github.com/user-attachments/assets/128ec166-5549-4ddc-b1ac-735537dd5698" />


### 백엔드 트러블 슈팅
---
| 1. 이미지 저장 및 경로 설정 이슈 |
|:---|
| **발생 배경** <br/> 이미지를 DB에 직접 저장하는 대신 문자열(Base64) 형태로 저장하고, 실제 파일은 로컬 디렉토리에 저장하는 방식을 적용하고자 함. 그러나, 저장 경로 지정 과정에서 프로젝트 내부와 로컬 디렉토리 간 경로 매핑이 필요. <br/> **문제 원인**
<br/>
- 리소스 폴더(`/resources`) 내 이미지 디렉토리와 로컬 저장 경로(`C:` 드라이브) 간의 일관성 있는 경로 설정 부재.
- MultipartFile 업로드 시 MediaType 설정 누락으로 파일 업로드 처리 오류 발생.
<br/>
**해결 방안**
<br/>
- `/resources/images` 디렉토리를 생성하여 내부 저장소 경로를 지정.
- 로컬 저장소(`C:/images`)에도 동일한 구조를 생성해 운영 환경과 로컬 환경의 경로 동기화.
- Spring Controller에서 `MultipartFile` 타입으로 수신하고, `MediaType`을 명시적으로 설정하여 이미지 처리 호환성 확보.
<br/> | 
| 설명 |
|이미지2 |
|설명2|
|이미지3|
|설명3|

### 개선사항

### 팀원 소개

