#!/usr/bin/env python3
"""
색상 채널 테스트 스크립트
"""

import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from modules.camera_handler import CameraHandler
from utils.frame_processor import FrameProcessor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_color_channels():
    """색상 채널 테스트"""
    print("🎨 색상 채널 테스트 시작")
    print("=" * 60)
    
    try:
        # 카메라 핸들러 초기화
        camera = CameraHandler(
            lq_resolution=(320, 240),
            hq_resolution=(640, 480)
        )
        
        print(f"📷 카메라 타입: {camera.get_camera_type()}")
        
        # 프레임 프로세서 초기화
        processor = FrameProcessor(max_frame_size=50*1024, auto_quality_adjust=False)
        
        # 다양한 색상 포맷으로 프레임 캡처
        print("\n🔍 색상 채널 비교 테스트...")
        
        # ML 모델용 프레임 (BGR)
        frame_ml = camera.capture_frame_for_ml(high_quality=True)
        if frame_ml is not None:
            print(f"✅ ML용 프레임 (BGR): {frame_ml.shape}")
            
            # 색상 히스토그램 분석
            b_hist = cv2.calcHist([frame_ml], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([frame_ml], [1], None, [256], [0, 256])
            r_hist = cv2.calcHist([frame_ml], [2], None, [256], [0, 256])
            
            print(f"   B 채널 평균: {np.mean(b_hist):.2f}")
            print(f"   G 채널 평균: {np.mean(g_hist):.2f}")
            print(f"   R 채널 평균: {np.mean(r_hist):.2f}")
            
            # 테스트 이미지 저장
            cv2.imwrite(f"test_ml_bgr_{int(time.time())}.jpg", frame_ml)
            print(f"   💾 ML용 BGR 이미지 저장됨")
        
        # 사용자용 프레임 (RGB)
        frame_user = camera.capture_frame_for_user(high_quality=True)
        if frame_user is not None:
            print(f"✅ 사용자용 프레임 (RGB): {frame_user.shape}")
            
            # OpenCV는 BGR이므로 RGB로 변환해서 저장
            frame_user_bgr = cv2.cvtColor(frame_user, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"test_user_rgb_{int(time.time())}.jpg", frame_user_bgr)
            print(f"   💾 사용자용 RGB 이미지 저장됨")
        
        # Base64 인코딩 테스트
        print(f"\n📤 Base64 인코딩 테스트...")
        
        if frame_ml is not None:
            # ML용 (BGR 유지)
            b64_ml, size_ml = processor.encode_frame_base64(frame_ml, quality=80, convert_to_rgb=False)
            print(f"✅ ML용 Base64 인코딩: {size_ml}B")
            
            # 사용자용 (RGB 변환)
            b64_user, size_user = processor.encode_frame_base64(frame_ml, quality=80, convert_to_rgb=True)
            print(f"✅ 사용자용 Base64 인코딩 (BGR→RGB): {size_user}B")
            
            print(f"   크기 차이: {abs(size_ml - size_user)}B")
        
        # 실시간 색상 비교 (5초간)
        print(f"\n🎥 실시간 색상 비교 테스트 (5초)...")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            # ML용과 사용자용 동시 캡처
            frame_ml = camera.capture_frame_for_ml(high_quality=False)
            frame_user = camera.capture_frame_for_user(high_quality=False)
            
            if frame_ml is not None and frame_user is not None:
                frame_count += 1
                
                # 픽셀 값 차이 비교 (중앙 픽셀)
                center_y, center_x = frame_ml.shape[0]//2, frame_ml.shape[1]//2
                
                ml_pixel = frame_ml[center_y, center_x]
                user_pixel = frame_user[center_y, center_x]
                
                if frame_count % 30 == 0:  # 30프레임마다 출력
                    print(f"   프레임 {frame_count}: ML(BGR)={ml_pixel}, User(RGB)={user_pixel}")
            
            time.sleep(0.1)
        
        # 정리
        camera.cleanup()
        
        print(f"\n✅ 색상 채널 테스트 완료!")
        print(f"총 {frame_count}프레임 처리됨")
        
        print(f"\n📋 사용법 가이드:")
        print(f"- ML 모델용: camera.capture_frame_for_ml() → BGR")
        print(f"- 사용자용: camera.capture_frame_for_user() → RGB") 
        print(f"- Base64 인코딩: processor.encode_frame_base64(frame, convert_to_rgb=True)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 색상 채널 테스트 실패: {e}")
        return False

def compare_yolo_compatibility():
    """YOLOv5 호환성 테스트"""
    print("🤖 YOLOv5 호환성 테스트")
    print("=" * 60)
    
    try:
        camera = CameraHandler(
            lq_resolution=(320, 240),
            hq_resolution=(640, 640)  # YOLO 입력 크기
        )
        
        # 프레임 캡처
        frame = camera.capture_frame_for_ml(high_quality=True)
        if frame is None:
            print("❌ 프레임 캡처 실패")
            return False
        
        print(f"✅ 원본 프레임: {frame.shape} (BGR)")
        
        # YOLOv5 전처리 시뮬레이션
        print(f"\n🔄 YOLOv5 전처리 시뮬레이션...")
        
        # 1. 정규화 (0-1)
        frame_normalized = frame.astype(np.float32) / 255.0
        print(f"   정규화: {frame_normalized.shape}, 범위: {frame_normalized.min():.3f}-{frame_normalized.max():.3f}")
        
        # 2. HWC → CHW 변환
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))
        print(f"   CHW 변환: {frame_chw.shape}")
        
        # 3. 배치 차원 추가
        frame_batch = np.expand_dims(frame_chw, axis=0)
        print(f"   배치 추가: {frame_batch.shape}")
        
        # 4. 색상 채널별 분석
        print(f"\n📊 색상 채널 분석:")
        for i, channel in enumerate(['Blue', 'Green', 'Red']):
            channel_data = frame_batch[0, i]
            print(f"   {channel} 채널: 평균={np.mean(channel_data):.3f}, 표준편차={np.std(channel_data):.3f}")
        
        # 테스트 이미지 저장
        test_filename = f"yolo_input_test_{int(time.time())}.jpg"
        cv2.imwrite(test_filename, frame)
        print(f"\n💾 YOLOv5 입력용 테스트 이미지: {test_filename}")
        
        camera.cleanup()
        
        print(f"\n✅ YOLOv5 호환성 테스트 완료!")
        print(f"🔍 결론: BGR 포맷으로 정상 처리 가능")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv5 호환성 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🎨 라즈베리파이 카메라 색상 채널 테스트")
    print("=" * 60)
    
    tests = [
        ("색상 채널 테스트", test_color_channels),
        ("YOLOv5 호환성 테스트", compare_yolo_compatibility),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n🧪 {name} 실행 중...")
            result = test_func()
            results.append((name, result))
            
            if result:
                print(f"✅ {name} 성공")
            else:
                print(f"❌ {name} 실패")
                
        except Exception as e:
            print(f"❌ {name} 오류: {e}")
            results.append((name, False))
        
        input("\nEnter를 눌러 다음 테스트로 진행하세요...")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("🏁 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 {len(results)}개 테스트 중 {passed}개 성공 ({passed/len(results)*100:.0f}%)")

if __name__ == "__main__":
    main()