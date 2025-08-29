#!/usr/bin/env python3
"""
Pi Camera v3 AF(Auto Focus) 테스트 스크립트
"""

import sys
import time
import cv2
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from modules.camera_handler import CameraHandler

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_autofocus():
    """AF 기능 종합 테스트"""
    print("🔍 Pi Camera v3 Auto Focus 테스트 시작")
    print("=" * 60)
    
    try:
        # 카메라 핸들러 초기화
        camera = CameraHandler(
            lq_resolution=(320, 240),
            hq_resolution=(1024, 768)
        )
        
        print(f"📷 카메라 타입: {camera.get_camera_type()}")
        
        # AF 정보 확인
        focus_info = camera.get_focus_info()
        print(f"🎯 AF 사용 가능: {focus_info['af_available']}")
        
        if not focus_info['af_available']:
            print("❌ AF를 사용할 수 없습니다. Pi Camera v3인지 확인하세요.")
            return False
        
        print(f"   현재 AF 모드: {focus_info['af_mode']}")
        print(f"   AF 상태: {focus_info['af_state']}")
        print(f"   렌즈 위치: {focus_info['lens_position']}")
        
        # 다양한 AF 모드 테스트
        af_modes = ["continuous", "auto", "manual"]
        
        for mode in af_modes:
            print(f"\n🔧 {mode} 모드 테스트 중...")
            camera.set_focus_mode(mode)
            time.sleep(2)  # 모드 변경 대기
            
            # 현재 상태 확인
            info = camera.get_focus_info()
            print(f"   AF 모드: {info['af_mode']}, 상태: {info['af_state']}, 렌즈: {info['lens_position']}")
            
            # 프레임 캡처 테스트
            frame = camera.capture_frame(high_quality=True)
            if frame is not None:
                print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
                
                # 이미지 선명도 측정 (Laplacian variance)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"   📊 이미지 선명도: {sharpness:.2f}")
                
                # 테스트 이미지 저장
                test_filename = f"af_test_{mode}_{int(time.time())}.jpg"
                cv2.imwrite(test_filename, frame)
                print(f"   💾 테스트 이미지 저장: {test_filename}")
            else:
                print("   ❌ 프레임 캡처 실패")
        
        # 수동 초점 거리 테스트
        print(f"\n🎛️ 수동 초점 거리 테스트...")
        focus_distances = [0.0, 2.5, 5.0, 10.0]  # 무한대 ~ 가까이
        
        for distance in focus_distances:
            print(f"   초점 거리 {distance} 설정 중...")
            camera.set_focus_distance(distance)
            time.sleep(2)
            
            frame = camera.capture_frame(high_quality=True)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"   📊 거리 {distance} 선명도: {sharpness:.2f}")
        
        # AF 트리거 테스트
        print(f"\n🔄 AF 트리거 테스트...")
        camera.set_focus_mode("auto")
        
        for i in range(3):
            print(f"   AF 트리거 {i+1}/3...")
            camera.trigger_autofocus()
            time.sleep(1)
            
            info = camera.get_focus_info()
            print(f"   상태: {info['af_state']}, 렌즈: {info['lens_position']}")
        
        # 최종 연속 AF 모드로 복원
        print(f"\n🔄 연속 AF 모드로 복원...")
        camera.set_focus_mode("continuous")
        
        # 정리
        camera.cleanup()
        
        print(f"\n✅ AF 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AF 테스트 실패: {e}")
        return False

def interactive_af_test():
    """대화형 AF 테스트"""
    print("🎮 대화형 AF 테스트 시작")
    print("명령어:")
    print("  c: Continuous AF")
    print("  a: Auto AF")  
    print("  m: Manual AF")
    print("  t: AF Trigger")
    print("  0-9: 수동 초점 거리 (0=무한대, 9=가까이)")
    print("  i: 현재 AF 정보 출력")
    print("  s: 현재 프레임 저장")
    print("  q: 종료")
    print("=" * 60)
    
    try:
        camera = CameraHandler(
            lq_resolution=(320, 240),
            hq_resolution=(1024, 768)
        )
        
        if not camera.get_focus_info()['af_available']:
            print("❌ AF를 사용할 수 없습니다.")
            return False
        
        while True:
            command = input("\n명령어 입력: ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'c':
                camera.set_focus_mode("continuous")
                print("✅ Continuous AF 모드 설정")
            elif command == 'a':
                camera.set_focus_mode("auto")
                print("✅ Auto AF 모드 설정")
            elif command == 'm':
                camera.set_focus_mode("manual")
                print("✅ Manual AF 모드 설정")
            elif command == 't':
                camera.trigger_autofocus()
                print("✅ AF 트리거 실행")
            elif command.isdigit():
                distance = float(command)
                camera.set_focus_distance(distance)
                print(f"✅ 수동 초점 거리 {distance} 설정")
            elif command == 'i':
                info = camera.get_focus_info()
                print(f"📊 AF 정보:")
                print(f"   모드: {info['af_mode']}")
                print(f"   상태: {info['af_state']}")
                print(f"   렌즈 위치: {info['lens_position']}")
            elif command == 's':
                frame = camera.capture_frame(high_quality=True)
                if frame is not None:
                    filename = f"manual_test_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    # 선명도 측정
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    print(f"💾 이미지 저장: {filename}")
                    print(f"📊 선명도: {sharpness:.2f}")
                else:
                    print("❌ 프레임 캡처 실패")
            else:
                print("❓ 알 수 없는 명령어")
            
            # 짧은 대기
            time.sleep(0.1)
        
        camera.cleanup()
        print("✅ 대화형 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 대화형 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🔍 Pi Camera v3 AF 테스트 도구")
    print("=" * 60)
    
    while True:
        print("\n테스트 모드 선택:")
        print("1. 자동 AF 테스트")
        print("2. 대화형 AF 테스트")
        print("3. 종료")
        
        choice = input("선택 (1-3): ").strip()
        
        if choice == '1':
            test_autofocus()
        elif choice == '2':
            interactive_af_test()
        elif choice == '3':
            print("👋 AF 테스트 도구를 종료합니다.")
            break
        else:
            print("❓ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()