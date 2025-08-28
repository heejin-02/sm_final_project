#!/bin/bash

# 라즈베리파이 카메라 시스템 설치 스크립트

echo "🔧 라즈베리파이 카메라 시스템 설치 시작"
echo "=================================================="

# 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo apt update && sudo apt upgrade -y

# 필요한 시스템 패키지 설치
echo "📦 시스템 라이브러리 설치..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    python3-numpy \
    python3-picamera2

# Python 가상환경 생성
echo "🐍 Python 가상환경 생성..."
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
echo "📦 Python 패키지 설치..."
pip install --upgrade pip
pip install -r requirements.txt

# 디렉토리 생성
echo "📁 필요한 디렉토리 생성..."
mkdir -p /home/pi/camera_backup
mkdir -p data/metadata
mkdir -p data/detections

# 카메라 모듈 활성화
echo "📷 카메라 모듈 설정..."
sudo raspi-config nonint do_camera 0

# 환경 변수 파일 생성
echo "⚙️ 환경 설정 파일 생성..."
cat > .env << EOF
# 카메라 설정
CAMERA_ID=cam_001
GH_IDX=74
SERVER_HOST=192.168.219.47

# 네트워크 설정 (필요시 수정)
# CAMERA_ID=cam_002
# SERVER_HOST=192.168.1.100
EOF

# 서비스 파일 생성 (옵션)
echo "🔧 시스템 서비스 파일 생성..."
sudo tee /etc/systemd/system/camera-client.service > /dev/null << EOF
[Unit]
Description=Camera Client for Insect Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/camera_system
ExecStart=/home/pi/camera_system/venv/bin/python camera_client.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 권한 설정
echo "🔐 권한 설정..."
sudo chown -R pi:pi /home/pi/camera_backup
chmod +x test_camera.py

# 서비스 등록 (아직 시작하지 않음)
sudo systemctl daemon-reload
sudo systemctl enable camera-client.service

echo "✅ 설치 완료!"
echo "=================================================="
echo ""
echo "🚀 사용 방법:"
echo "1. 테스트 실행: python test_camera.py"
echo "2. 카메라 클라이언트 실행: python camera_client.py"
echo "3. 시스템 서비스로 실행: sudo systemctl start camera-client"
echo ""
echo "⚙️ 설정 파일: .env"
echo "📁 백업 위치: /home/pi/camera_backup"
echo ""
echo "🔍 문제 해결:"
echo "- 로그 확인: journalctl -u camera-client -f"
echo "- 서비스 상태: systemctl status camera-client"
echo "- 카메라 테스트: libcamera-hello"
echo ""
echo "⚠️ 주의사항:"
echo "- .env 파일에서 CAMERA_ID와 SERVER_HOST를 설정하세요"
echo "- ML API 서버(8003 포트)가 실행 중인지 확인하세요"
echo "- 같은 와이파이 네트워크에 연결되어 있는지 확인하세요"