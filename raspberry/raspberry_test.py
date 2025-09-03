#!/usr/bin/env python3
"""
라즈베리파이 환경 테스트 스크립트
모든 구성 요소가 제대로 설치되었는지 확인
"""

import sys
import platform

print("=" * 50)
print("🍓 라즈베리파이 환경 테스트")
print("=" * 50)

# 1. Python 버전 확인
print(f"\n✅ Python 버전: {sys.version}")
print(f"✅ 플랫폼: {platform.platform()}")
print(f"✅ 아키텍처: {platform.machine()}")

# 2. OpenCV 확인
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
    
    # 카메라 테스트
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ 카메라: 정상 작동 (해상도: {frame.shape})")
        else:
            print("❌ 카메라: 프레임 읽기 실패")
        cap.release()
    else:
        print("❌ 카메라: 연결 실패")
except ImportError as e:
    print(f"❌ OpenCV 설치 필요: {e}")
except Exception as e:
    print(f"⚠️ 카메라 테스트 실패: {e}")

# 3. PyTorch 확인
try:
    import torch
    import torchvision
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ TorchVision: {torchvision.__version__}")
    print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
    
    # 간단한 텐서 연산 테스트
    x = torch.randn(1, 3, 224, 224)
    print(f"✅ 텐서 생성: {x.shape}")
except ImportError as e:
    print(f"❌ PyTorch 설치 필요: {e}")

# 4. NumPy 확인
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy 설치 필요: {e}")

# 5. 네트워크 패키지 확인
try:
    import aiohttp
    print(f"✅ aiohttp: {aiohttp.__version__}")
except ImportError as e:
    print(f"❌ aiohttp 설치 필요: {e}")

try:
    import requests
    print(f"✅ requests: {requests.__version__}")
    
    # 네트워크 연결 테스트
    try:
        response = requests.get("http://www.google.com", timeout=5)
        print(f"✅ 인터넷 연결: 정상")
    except:
        print(f"⚠️ 인터넷 연결: 확인 필요")
except ImportError as e:
    print(f"❌ requests 설치 필요: {e}")

# 6. scikit-learn 확인
try:
    import sklearn
    print(f"✅ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn 설치 필요: {e}")

# 7. PIL 확인
try:
    from PIL import Image
    print(f"✅ Pillow: {Image.__version__}")
except ImportError as e:
    print(f"❌ Pillow 설치 필요: {e}")

# 8. 메모리 상태 확인
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n📊 메모리 상태:")
    print(f"   - 전체: {memory.total / 1024**3:.2f} GB")
    print(f"   - 사용 가능: {memory.available / 1024**3:.2f} GB")
    print(f"   - 사용률: {memory.percent}%")
except:
    # psutil이 없어도 기본 정보 출력
    import os
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            total = int(lines[0].split()[1]) / 1024 / 1024
            available = int(lines[2].split()[1]) / 1024 / 1024
            print(f"\n📊 메모리 상태:")
            print(f"   - 전체: {total:.2f} GB")
            print(f"   - 사용 가능: {available:.2f} GB")
    except:
        print("⚠️ 메모리 정보 확인 불가")

# 9. 디스크 공간 확인
import shutil
total, used, free = shutil.disk_usage("/")
print(f"\n💾 디스크 공간:")
print(f"   - 전체: {total // (2**30)} GB")
print(f"   - 사용: {used // (2**30)} GB")
print(f"   - 여유: {free // (2**30)} GB")

# 10. 모델 파일 확인
import os
model_path = os.path.expanduser("~/insect_detection/models/best_mobilenet_insect.pt")
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / 1024 / 1024
    print(f"\n✅ 모델 파일: 존재 ({size:.2f} MB)")
else:
    print(f"\n⚠️ 모델 파일 없음: {model_path}")

print("\n" + "=" * 50)

# 종합 결과
missing = []
if 'cv2' not in sys.modules:
    missing.append("opencv-python")
if 'torch' not in sys.modules:
    missing.append("torch")
if 'numpy' not in sys.modules:
    missing.append("numpy")
if 'aiohttp' not in sys.modules:
    missing.append("aiohttp")
if 'requests' not in sys.modules:
    missing.append("requests")
if 'sklearn' not in sys.modules:
    missing.append("scikit-learn")

if missing:
    print(f"❌ 설치 필요한 패키지:")
    for pkg in missing:
        print(f"   pip install {pkg}")
else:
    print("✅ 모든 필수 패키지 설치 완료!")
    print("🚀 라즈베리파이 실행 준비 완료!")

print("=" * 50)