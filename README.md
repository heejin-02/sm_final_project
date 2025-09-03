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
| 실시간 탐지 페이지 | 탐지 이력/통계 대시보드 |
|:---:|:---:|
| (이미지/영상 링크 자리) | (이미지/영상 링크 자리) |
| 제목 | 제목 |
| (이미지/영상 링크 자리) | (이미지/영상 링크 자리) |
| (이미지/영상 링크 자리) | (이미지/영상 링크 자리) |
| (이미지/영상 링크 자리) | (이미지/영상 링크 자리) |


### 🚀 주요 기능
- 🐛 **실시간 해충 탐지 (YOLOv5 + OpenCV)**  
- 🎥 탐지 시 **10초간 영상 자동 저장**  
- 📞 **전화 음성 알림 발송 (SignalWire API)**  
- 📊 **탐지 이력/발생률 통계** 제공 (Chart.js 시각화)  
- 🧠 **해충 맞춤 방제 챗봇** (LangChain + ChromaDB 기반)  
- 🧑‍🌾 **고령자 친화형 UI** (큰 글씨, 다크모드, 음성 버튼 지원)  



### 서비스 아키텍처
---
Webcam에서 촬영된 영상은 YOLOv5 감지 서버로 전송되며, 서버에서는 10초 단위로 영상을 저장합니다. 감지 서버는 동시에 여러 작업을 수행하는데, WebSocket을 통해 프론트엔드에 실시간 전화알림을 보내고, SignalWire TTS를 이용해 전화로 음성 알림을 제공합니다. 또한 GPT를 활용하여 벌레 리포트를 생성하며, LangChain과 ChromaDB를 활용해 방제 관련 챗봇 응답도 제공합니다.

<img width="1418" height="800" alt="image" src="https://github.com/user-attachments/assets/128ec166-5549-4ddc-b1ac-735537dd5698" />


### 백엔드 트러블 슈팅
---
| 트러블슈팅 |
|:---:|
| 이미지1|
| 설명 |
|이미지2 |
|설명2|
|이미지3|
|설명3|

### 개선사항

### 팀원 소개

