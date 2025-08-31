#!/bin/bash
# 라즈베리파이 4 환경 설정 스크립트
# Raspberry Pi OS (64-bit) 권장

echo "🍓 라즈베리파이 해충 탐지 시스템 설치 시작..."

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. 시스템 의존성 설치
echo "🔧 시스템 의존성 설치..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    python3-pyqt5 \
    libjasper-dev

# 3. Python 가상환경 생성
echo "🐍 Python 가상환경 생성..."
python3 -m venv ~/insect_env
source ~/insect_env/bin/activate

# 4. pip 업그레이드
pip install --upgrade pip setuptools wheel

# 5. NumPy 먼저 설치 (의존성 문제 해결)
echo "📊 NumPy 설치..."
pip install numpy==1.24.3

# 6. OpenCV 설치 (라즈베리파이 최적화 버전)
echo "📷 OpenCV 설치..."
pip install opencv-python==4.8.1.78

# 7. PyTorch 설치 (ARM64용)
echo "🔥 PyTorch 설치 (CPU 버전)..."
# 라즈베리파이 4용 PyTorch
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# 8. 나머지 패키지 설치
echo "📦 추가 패키지 설치..."
pip install \
    Pillow==10.0.0 \
    scikit-learn==1.3.0 \
    scipy==1.11.2 \
    aiohttp==3.8.5 \
    requests==2.31.0 \
    python-dateutil==2.8.2

# 9. 메모리 스왑 설정 (RAM 부족 대비)
echo "💾 스왑 메모리 설정..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 10. 카메라 활성화
echo "📹 카메라 모듈 활성화..."
sudo raspi-config nonint do_camera 0

# 11. GPU 메모리 할당 (카메라용)
echo "🎮 GPU 메모리 설정..."
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# 12. 모델 파일 다운로드 위치 생성
echo "📁 작업 디렉토리 생성..."
mkdir -p ~/insect_detection/models
mkdir -p ~/insect_detection/offline_queue
mkdir -p ~/insect_detection/logs

# 13. 서비스 자동 시작 설정 (선택사항)
echo "🚀 자동 시작 서비스 생성..."
cat > ~/insect_detection/insect_monitor.service << EOL
[Unit]
Description=Insect Detection Monitor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/insect_detection
Environment="PATH=/home/pi/insect_env/bin"
ExecStart=/home/pi/insect_env/bin/python /home/pi/insect_detection/raspberry_integrated_v2.py
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# 서비스 설치 (선택사항)
# sudo cp ~/insect_detection/insect_monitor.service /etc/systemd/system/
# sudo systemctl enable insect_monitor.service
# sudo systemctl start insect_monitor.service

echo "✅ 설치 완료!"
echo "📝 다음 단계:"
echo "1. MobileNet 모델 파일 복사: ~/insect_detection/models/best_mobilenet_insect.pt"
echo "2. 코드 파일 복사: ~/insect_detection/raspberry_integrated_v2.py"
echo "3. 설정 수정: Spring Boot URL, ML Server URL"
echo "4. 실행: python ~/insect_detection/raspberry_integrated_v2.py"
echo ""
echo "⚠️ 재부팅 필요: sudo reboot"