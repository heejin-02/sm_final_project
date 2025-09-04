#!/bin/bash

# 전화 알림 서비스 시작 스크립트
echo "📞 전화 알림 서비스 시작 (포트 8005)..."

cd ../phone_service
source venv/bin/activate
python main.py