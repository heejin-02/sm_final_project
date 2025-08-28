# 라즈베리파이 카메라 시스템

실시간 해충 탐지를 위한 라즈베리파이 4B + Pi Camera v3 기반 멀티 카메라 시스템

## 🏗️ 시스템 아키텍처

```
라즈베리파이 카메라 → WebSocket → ML API 서버 → Spring Boot API → 알림/DB
```

### 주요 특징
- **듀얼 스트림**: LQ(320x240) 움직임 감지 + HQ(1024x768) 해충 탐지
- **자동 품질 조절**: 네트워크 상황에 따른 JPEG 압축률 자동 조절
- **실시간 처리**: WebSocket을 통한 저지연 영상 전송
- **메타데이터 관리**: CSV 기반 탐지 결과 및 추적 정보 저장
- **로컬 백업**: 네트워크 장애 대비 로컬 저장

## 📦 설치 방법

### 1. 자동 설치
```bash
git clone <repository>
cd raspberry
chmod +x setup.sh
./setup.sh
```

### 2. 수동 설치
```bash
# 시스템 패키지 설치
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libopencv-dev python3-opencv python3-picamera2

# Python 가상환경
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 카메라 모듈 활성화
sudo raspi-config nonint do_camera 0
```

## ⚙️ 설정

### 환경 변수 (.env)
```bash
# 카메라 식별
CAMERA_ID=cam_001          # 각 라즈베리파이마다 고유 ID
GH_IDX=74                  # 온실 인덱스

# 네트워크
SERVER_HOST=192.168.219.47 # ML API 서버 IP
```

### 카메라 설정 (config.py)
```python
# 해상도 설정
lq_resolution: (320, 240)    # 움직임 감지용
hq_resolution: (1024, 768)   # 탐지용

# 프레임 설정
lq_fps: 10                   # LQ 스트림 FPS
hq_fps: 5                    # HQ 스트림 FPS

# 압축 설정
jpeg_quality_low: 30         # 낮은 품질 (대역폭 절약)
jpeg_quality_high: 80        # 높은 품질 (탐지용)
max_frame_size: 50 * 1024    # 최대 프레임 크기 (50KB)
```

## 🚀 사용 방법

### 1. 테스트 실행
```bash
python test_camera.py
```

### 2. 직접 실행
```bash
python camera_client.py
```

### 3. 시스템 서비스 실행
```bash
sudo systemctl start camera-client
sudo systemctl enable camera-client  # 부팅시 자동 시작
```

## 📊 모니터링

### 로그 확인
```bash
# 실시간 로그
journalctl -u camera-client -f

# 서비스 상태
systemctl status camera-client
```

### 카메라 테스트
```bash
# 카메라 하드웨어 테스트
libcamera-hello

# 영상 확인
libcamera-vid -t 10000 --width 1024 --height 768 -o test.h264
```

## 🔧 API 엔드포인트

### WebSocket
- `ws://{server_host}:8003/ws/camera` - 카메라 연결

### REST API
- `GET /api/camera/stats` - 카메라 통계 조회

## 📁 파일 구조

```
raspberry/
├── camera_client.py       # 메인 클라이언트
├── config.py             # 설정 관리
├── requirements.txt      # Python 의존성
├── setup.sh             # 자동 설치 스크립트
├── test_camera.py       # 테스트 스크립트
└── utils/
    ├── motion_detector.py    # 움직임 감지
    └── frame_processor.py    # 프레임 처리/압축
```

## 🔍 문제 해결

### 일반적인 문제
1. **카메라 인식 안됨**
   ```bash
   sudo raspi-config  # Camera 활성화
   sudo reboot
   ```

2. **네트워크 연결 실패**
   - ML API 서버(포트 8003) 실행 상태 확인
   - IP 주소 및 방화벽 설정 확인

3. **성능 저하**
   - 해상도 및 FPS 조절
   - JPEG 품질 설정 최적화

### 성능 최적화
- **메모리 분할**: `sudo raspi-config` > Advanced Options > Memory Split > 128
- **GPU 메모리**: `/boot/config.txt`에 `gpu_mem=128` 추가
- **오버클럭**: 냉각이 충분한 경우 CPU 속도 향상

## 🔒 보안 고려사항

- 네트워크 접근 제한 (동일 서브넷만)
- 인증서를 통한 HTTPS/WSS 사용 권장
- 정기적인 시스템 업데이트

## 📈 모니터링 메트릭

- FPS (초당 프레임)
- 프레임 크기 (평균/최대)
- 네트워크 대역폭 사용량
- 움직임 감지율
- 해충 탐지 정확도

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request