"""
라즈베리파이 카메라 시스템 테스트 스크립트
"""

import asyncio
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from camera_client import CameraClient
from utils.motion_detector import MotionDetector
from utils.frame_processor import FrameProcessor

def test_motion_detector():
    """움직임 감지 테스트"""
    print("=" * 50)
    print("움직임 감지 테스트 시작")
    print("=" * 50)
    
    motion_detector = MotionDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    print("웹캠이 준비되었습니다. 'q'를 누르면 종료됩니다.")
    print("화면 앞에서 손을 흔들어 움직임을 감지해보세요.")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 움직임 감지
        motion_detected, motion_areas = motion_detector.detect_motion(frame)
        
        if motion_detected:
            detection_count += 1
            print(f"[Frame {frame_count}] 움직임 감지! 영역 수: {len(motion_areas)}")
            
            # 움직임 영역 그리기
            frame = motion_detector.draw_motion_areas(frame, motion_areas)
        
        # 상태 표시
        status = "MOTION DETECTED!" if motion_detected else "No Motion"
        color = (0, 255, 0) if motion_detected else (255, 255, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow("Motion Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n테스트 완료!")
    print(f"총 프레임 수: {frame_count}")
    print(f"움직임 감지 횟수: {detection_count}")
    print(f"감지율: {detection_count/frame_count*100:.1f}%")
    
    return True

def test_frame_processor():
    """프레임 처리기 테스트"""
    print("=" * 50)
    print("프레임 처리기 테스트 시작")
    print("=" * 50)
    
    processor = FrameProcessor(max_frame_size=30*1024, auto_quality_adjust=True)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("프레임 압축 테스트 중... 10초간 진행됩니다.")
    
    start_time = time.time()
    frame_count = 0
    total_size = 0
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 자동 품질 조절로 인코딩
        base64_data, size, quality = processor.auto_adjust_quality(frame)
        total_size += size
        
        if frame_count % 30 == 0:  # 30프레임마다 출력
            stats = processor.get_bandwidth_stats()
            print(f"프레임 {frame_count}: 크기 {size}B, 품질 {quality}%, 평균 {stats['avg_frame_size']:.0f}B")
    
    cap.release()
    
    # 결과 출력
    avg_size = total_size / frame_count
    avg_fps = frame_count / 10
    bandwidth = avg_size * avg_fps / 1024  # KB/s
    
    print(f"\n테스트 완료!")
    print(f"총 프레임 수: {frame_count}")
    print(f"평균 프레임 크기: {avg_size:.0f}B")
    print(f"평균 FPS: {avg_fps:.1f}")
    print(f"예상 대역폭: {bandwidth:.1f} KB/s")
    
    return True

def test_config():
    """설정 테스트"""
    print("=" * 50)
    print("설정 테스트 시작")
    print("=" * 50)
    
    from config import get_config
    
    config = get_config()
    
    print(f"카메라 ID: {config.camera_id}")
    print(f"온실 인덱스: {config.gh_idx}")
    print(f"서버 주소: {config.server_host}:{config.server_port}")
    print(f"LQ 해상도: {config.lq_resolution}")
    print(f"HQ 해상도: {config.hq_resolution}")
    print(f"최대 프레임 크기: {config.max_frame_size}B")
    print(f"자동 품질 조절: {config.auto_quality_adjust}")
    
    return True

async def test_websocket_connection():
    """웹소켓 연결 테스트"""
    print("=" * 50)
    print("웹소켓 연결 테스트 시작")
    print("=" * 50)
    
    import websockets
    import json
    from config import get_config
    
    config = get_config()
    uri = f"ws://{config.server_host}:{config.server_port}{config.websocket_endpoint}"
    
    try:
        print(f"연결 시도: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ 웹소켓 연결 성공!")
            
            # 초기화 메시지 전송
            init_message = {
                "type": "camera_init",
                "camera_id": "test_camera",
                "gh_idx": 74,
                "config": {
                    "lq_resolution": [320, 240],
                    "hq_resolution": [1024, 768],
                    "lq_fps": 10
                }
            }
            
            await websocket.send(json.dumps(init_message))
            print("📤 초기화 메시지 전송")
            
            # 응답 대기
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 서버 응답: {response_data}")
            
            # ping 테스트
            ping_message = {"type": "ping"}
            await websocket.send(json.dumps(ping_message))
            
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"🏓 Pong: {pong_data}")
            
        print("✅ 웹소켓 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 웹소켓 연결 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔧 라즈베리파이 카메라 시스템 테스트")
    print("=" * 60)
    
    tests = [
        ("설정 테스트", test_config),
        ("움직임 감지 테스트", test_motion_detector),
        ("프레임 처리기 테스트", test_frame_processor),
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
    
    # 웹소켓 테스트 (비동기)
    print(f"\n🧪 웹소켓 연결 테스트 실행 중...")
    try:
        result = asyncio.run(test_websocket_connection())
        results.append(("웹소켓 연결 테스트", result))
    except Exception as e:
        print(f"❌ 웹소켓 테스트 오류: {e}")
        results.append(("웹소켓 연결 테스트", False))
    
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
    
    if passed == len(results):
        print("\n🎉 모든 테스트가 성공했습니다! 시스템이 준비되었습니다.")
    else:
        print(f"\n⚠️ {len(results) - passed}개 테스트가 실패했습니다. 문제를 해결해주세요.")

if __name__ == "__main__":
    main()